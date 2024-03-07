import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from functools import partial
import torch.nn.functional as F
from pcdet import device
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from pcdet.ops.bev_pool import bev_pool_ext
from pcdet.ops.box2map import box2map
import spconv.pytorch as spconv

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterSpeedHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.train_box = model_cfg.TRAIN_BOX
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5),
                            momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())
        self.add_module('speed_loss_func', torch.nn.L1Loss())
        self.add_module('speed_cls_loss_func', torch.nn.BCELoss())
        # self.add_module('speed_loss_func',loss_utils.)

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[0], feature_map_size[1])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] + 1 - (0 if self.train_box else 2)))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes
        ret_boxes_src = torch.concat([ret_boxes_src[:, :7], ret_boxes_src[:, -1:]], dim=-1)
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0,
                              max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        boxes = gt_boxes[:, :7].contiguous()

        boxes[:, :2] = (boxes[:, :2] - torch.from_numpy(self.point_cloud_range)[:2]) / torch.tensor(
            self.voxel_size[:2])[None, :]
        boxes[:, 3:5] = (boxes[:, 3:5]) / torch.tensor(self.voxel_size[:2])[None, :]
        boxes[:, 6][boxes[:, 6] < 0] += torch.pi
        if not self.train_box:
            speed = gt_boxes[:, 7:9].contiguous()
            speed = torch.cat([speed, torch.arange(1, boxes.shape[0] + 1)[:, None]], dim=-1)

            speed_map = torch.zeros([size * feature_map_stride for size in feature_map_size] + [speed.shape[-1]]).to(device)
            box2map.box2map_gpu(boxes.to(device), speed_map, speed.to(device))

            assert boxes[:,0].min()>0 and boxes[:,0].max()<1504 and boxes[:,1].min()>0 and boxes[:,1].max()<1504

        else:
            speed_map = None
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 0] * feature_map_size[1] + center_int[k, 1]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if ret_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, -1:]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src, speed_map

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
            'speed_map': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list, speed_map_list = [], [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)


                heatmap, ret_boxes, inds, mask, ret_boxes_src, speed_map = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.to('cpu'),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,

                )
                if not self.train_box:
                    try:
                        speed_map_list.append(speed_map.to(gt_boxes_single_head.device))
                    except:
                        pass
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0).to(device))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0).to(device))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0).to(device))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0).to(device))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0).to(device))
            if not self.train_box:
                ret_dict['speed_map'].append(torch.stack(speed_map_list, dim=0).to(device))
        return ret_dict

    def spatial_consistency_loss(self, speed_pred_coords, speed_pred, speed_gt):
        sc_loss = 0
        sc_loss_func = nn.L1Loss()
        B = speed_pred_coords[:, 0].max().item() + 1
        counter = 0
        for b in range(1,B):
            speed_pred_single_batch = speed_pred[speed_pred_coords[:, 0] == b]
            speed_gt_single_batch = speed_gt[speed_pred_coords[:, 0] == b]
            N = int(speed_gt_single_batch[:, -1].max().item())
            for n in range(1, N + 1):  # 0 denotes bg
                speed_pred_single_box = speed_pred_single_batch[speed_gt_single_batch[:, -1] == n]
                if speed_pred_single_box.shape[0]:  # avoid empty box causing nan
                    speed_pred_single_box_1 = speed_pred_single_box[:, None, :].repeat(1,
                                                                                       speed_pred_single_box.shape[0],
                                                                                       1)
                    speed_pred_single_box_2 = speed_pred_single_box[None, :, :].repeat(speed_pred_single_box.shape[0],
                                                                                       1, 1)
                    sc_loss += sc_loss_func(speed_pred_single_box_1, speed_pred_single_box_2)
            counter += torch.unique(speed_gt_single_batch[:, -1]).shape[0]

        return sc_loss / counter

    def temporal_consistency_loss(self, speed_pred_coords, speed_pred, speed_gt):
        sp_pred_tensor = spconv.SparseConvTensor(
            features=speed_pred,
            indices=speed_pred_coords.to(dtype=torch.int32),
            spatial_shape=[self.F,self.grid_size[1],self.grid_size[0]],  # easy to implement deconv
            batch_size=self.B
        )
        sp_pred = sp_pred_tensor.dense()
        sp_pred_mask = torch.zeros((sp_pred.shape[0],sp_pred.shape[2],sp_pred.shape[3],sp_pred.shape[4]),dtype=torch.bool).to(device)
        sp_pred_mask[speed_pred_coords[:,0],speed_pred_coords[:,1],speed_pred_coords[:,2],speed_pred_coords[:,3]] = True
        sp_pred = sp_pred.permute(0,3,4,2,1)
        sp_pred_mask = sp_pred_mask.permute(0,2,3,1)
        loss = 0
        mask1 = torch.logical_and(sp_pred_mask[...,1],sp_pred_mask[...,2])
        mask2 = torch.logical_and(sp_pred_mask[...,1],sp_pred_mask[...,3])
        mask3 = torch.logical_and(sp_pred_mask[...,2],sp_pred_mask[...,3])
        loss+=self.speed_loss_func(sp_pred[:,:,:,1,:][mask1],sp_pred[:,:,:,2,:][mask1])
        loss+=self.speed_loss_func(sp_pred[:,:,:,1,:][mask2],sp_pred[:,:,:,3,:][mask2])
        loss+=self.speed_loss_func(sp_pred[:,:,:,2,:][mask3],sp_pred[:,:,:,3,:][mask3])

        return loss

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        if self.train_box:
            for idx, pred_dict in enumerate(pred_dicts):
                pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
                hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
                hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

                target_boxes = target_dicts['target_boxes'][idx]
                pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

                reg_loss = self.reg_loss_func(
                    pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes[:,:,:8]
                )
                loc_loss = (reg_loss * reg_loss.new_tensor(
                    self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
                loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                loss += hm_loss + loc_loss
                tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
                tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
                if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):
                    batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                        pred_dict=pred_dict,
                        point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                        feature_map_stride=self.feature_map_stride
                    )  # (B, H, W, 7 or 9)

                    if 'iou' in pred_dict:
                        batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                        iou_loss = loss_utils.calculate_iou_loss_centerhead(
                            iou_preds=pred_dict['iou'],
                            batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                            mask=target_dicts['masks'][idx],
                            ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                        )
                        loss += iou_loss
                        tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                    if self.model_cfg.get('IOU_REG_LOSS', False):
                        iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                            batch_box_preds=batch_box_preds_for_iou,
                            mask=target_dicts['masks'][idx],
                            ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                        )
                        if target_dicts['masks'][idx].sum().item() != 0:
                            iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                            loss += iou_reg_loss
                            tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                        else:
                            loss += (batch_box_preds_for_iou * 0.).sum()
                            tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

            tb_dict['rpn_loss'] = loss.item()
            return loss, tb_dict


        else:
            B = pred_dicts[0]['coordinate_all'][:, 0].max() + 1
            FRAME = pred_dicts[0]['coordinate_all'][:, 1].max() + 1
            for idx, pred_dict in enumerate(pred_dicts):

                speed_map = target_dicts['speed_map'][idx]
                speed_pred = pred_dict['speed_1st']
                abs_speed_map = torch.norm(speed_map[:,:,:,:2], dim=-1, p=2)
                coordinate_all = pred_dict['coordinate_all'].long()
                pillar_coordinates = torch.zeros((coordinate_all.shape[0], 3)).long().to(device)

                pillar_coordinates[:, 1:] = coordinate_all[:, 2:]
                pillar_coordinates[:, 0] = coordinate_all[:, 0] * FRAME + coordinate_all[:, 1]
                speed_gt = speed_map[pillar_coordinates[:, 0], pillar_coordinates[:, 1], pillar_coordinates[:, 2]]
                gt_mask = speed_gt[:, -1] > 0
                is_moving_pred = pred_dict['is_moving'].float()
                abs_gt = torch.norm(speed_gt[:, :2], dim=-1, p=2)
                is_moving_label = torch.zeros(abs_gt.shape[0]).to(device)
                moving_mask = abs_gt > 0.3
                static_mask = (abs_gt < 0.1) * gt_mask
                is_moving_label[moving_mask] = 1

                speed_map_compressed = speed_map.reshape((B, FRAME, speed_map.shape[1], speed_map.shape[2], -1))
                speed_map_compressed_inds = (speed_map_compressed[:, :, :, :, -1] > 0).sum(dim=1)
                speed_map_compressed_mask = speed_map_compressed_inds > 0
                speed_map_compressed = torch.sum(speed_map_compressed[:, :, :, :, :2], dim=1)[
                                       speed_map_compressed_mask] / speed_map_compressed_inds[:, :, :, None][
                                       speed_map_compressed_mask]
                speed_map_compressed_pred = pred_dict['speed_compressed'][speed_map_compressed_mask]


                speed_loss = self.speed_loss_func(speed_pred[gt_mask], speed_gt[:, :-1][gt_mask])
                speed_compressed_loss = self.speed_loss_func(speed_map_compressed_pred,speed_map_compressed)
                speed_cls_loss = self.speed_cls_loss_func(is_moving_pred[gt_mask], is_moving_label[gt_mask])

                spatial_consistency_loss = self.spatial_consistency_loss(pillar_coordinates, speed_pred, speed_gt)
                temporal_consistency_loss = self.temporal_consistency_loss(coordinate_all, speed_pred, speed_gt)

                motion_mask = abs_speed_map > self.model_cfg.LOSS_CONFIG.SPEED_THRESHOLD

                target_boxes = target_dicts['target_boxes'][idx]
                pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

                print('speed_ls:{:.3f} speed_compre_ls:{:.3d} cls_loss:{:.3f}  spa_consis_ls:{:.3f}  temp_consis_ls:{:.3f}'.format(speed_loss,speed_compressed_loss,speed_cls_loss,spatial_consistency_loss,temporal_consistency_loss))
                loss += speed_loss+speed_compressed_loss+speed_cls_loss+spatial_consistency_loss+temporal_consistency_loss

                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                    iou_loss = loss_utils.calculate_iou_loss_centerhead(
                        iou_preds=pred_dict['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

            tb_dict['rpn_loss'] = loss.item()
            return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'],
                                                          1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(
                        pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):

        pred_dicts = []
        if 'pillar_coords' in data_dict:
            self.B = data_dict['pillar_coords'][:,0].max().item()+1
            self.F = data_dict['pillar_coords'][:,1].max().item()+1
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        for head in self.heads_list:
            pred_dict = head(x)
            pred_dicts.append(pred_dict)

        gt_boxes = data_dict['gt_boxes']

        if not self.train_box:
            FRAME = 4
            num_gt_boxes = torch.tensor([(gt_box[:, -2] == i).sum() for gt_box in gt_boxes for i in range(FRAME)])
            gt_boxes_new = torch.zeros(
            (data_dict['batch_size'], num_gt_boxes.max(), gt_boxes.shape[-1] - 1))  # remove vx,vy,frame_id
            for b in range(gt_boxes.shape[0]):
                for f in range(FRAME):
                    temp = gt_boxes[b][gt_boxes[b][:, -2] == f]
                    temp = torch.concat([temp[:, :-2], temp[:, -1:]], dim=-1)
                    gt_boxes_new[b * FRAME + f, :temp.shape[0]] = temp
        else:
            gt_boxes_new = gt_boxes
        if self.training:
            target_dict = self.assign_targets(
                gt_boxes_new, feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )

            inds = target_dict['inds'][0]

            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )
            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
