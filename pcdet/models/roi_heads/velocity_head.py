from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.functional as F
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.velocitynet_utils import build_transformer, PointNet2, MLP, build_voxel_sampler
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet import device
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu

class ProposalTargetLayerMPPNet(ProposalTargetLayer):
    def __init__(self, roi_sampler_cfg):
        super().__init__(roi_sampler_cfg=roi_sampler_cfg)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """

        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels, \
            = self.sample_rois_for_mppnet(batch_dict=batch_dict)

        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError
        num_roi = 128
        targets_dict = {'rois': batch_rois[:, :num_roi, :], 'gt_of_rois': batch_gt_of_rois[:, :num_roi, :],
                        'gt_iou_of_rois': batch_roi_ious, 'roi_scores': batch_roi_scores,
                        'roi_labels': batch_roi_labels, 'reg_valid_mask': reg_valid_mask[:, :num_roi],
                        'rcnn_cls_labels': batch_cls_labels[:, :num_roi], 'trajectory_rois': batch_rois,
                        # 'trajectory_rois': batch_trajectory_rois,
                        # 'valid_length': batch_valid_length,
                        }

        return targets_dict

    def sample_rois_for_mppnet(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
        """
        FRAME = batch_dict['num_points_all'].shape[1]
        batch_size = batch_dict['batch_size']

        # trajectory_rois = batch_dict['trajectory_rois']
        proposals_list = batch_dict['proposals_list']

        # batch_trajectory_rois = proposals_list.new_zeros(batch_size, proposals_list.shape[1],
        #                                                   self.roi_sampler_cfg.ROI_PER_IMAGE, proposals_list.shape[-1])

        # valid_length = batch_dict['valid_length']
        # batch_valid_length = trajectory_rois.new_zeros(
        #     (batch_size, batch_dict['trajectory_rois'].shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE))
        batch_rois_list = []
        batch_gt_of_rois_list = []
        batch_roi_ious_list = []
        batch_roi_scores_list = []
        batch_roi_labels_list = []

        for cur_frame_idx in range(FRAME):
            # cur_frame_idx = 0

            rois = batch_dict['proposals_list'][:, cur_frame_idx, :, :]
            roi_scores = batch_dict['roi_scores'][:, cur_frame_idx]
            roi_labels = batch_dict['roi_labels'][:, cur_frame_idx]
            # gt_boxes = batch_dict['gt_boxes'][batch_dict['gt_boxes'][:, :, -2] == cur_frame_idx + 1].unsqueeze(0)

            code_size = rois.shape[-1]
            batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
            batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, batch_dict['gt_boxes'].shape[-1])
            batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
            batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
            batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

            for index in range(batch_size):

                # cur_trajectory_rois = proposals_list[index]

                cur_roi,  cur_roi_labels, cur_roi_scores = rois[index], roi_labels[index], \
                    roi_scores[index]
                cur_gt = batch_dict['gt_boxes'][index][batch_dict['gt_boxes'][index][:,-2]==cur_frame_idx+1]
                if 'valid_length' in batch_dict.keys():
                    cur_valid_length = valid_length[index].to(device)

                k = cur_gt.__len__() - 1
                while k > 0 and cur_gt[k].sum() == 0:
                    k -= 1

                cur_gt = cur_gt[:k + 1]
                cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

                if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                    max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                        rois=cur_roi, roi_labels=cur_roi_labels.long(),
                        gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                    )

                else:
                    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi.cuda(), cur_gt.cuda()[:, 0:7]).to(device)  # (M, N)
                    max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

                sampled_inds, fg_inds, bg_inds = self.subsample_rois(max_overlaps=max_overlaps)

                batch_roi_labels[index] = cur_roi_labels[sampled_inds.long()]

                if self.roi_sampler_cfg.get('USE_ROI_AUG', False):

                    fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(cur_roi[fg_inds], cur_gt[gt_assignment[fg_inds]],
                                                                    max_overlaps[fg_inds],
                                                                    aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES)
                    bg_rois = cur_roi[bg_inds]
                    bg_iou3d = max_overlaps[bg_inds]

                    batch_rois[index] = torch.cat([fg_rois, bg_rois], 0)
                    batch_roi_ious[index] = torch.cat([fg_iou3d, bg_iou3d], 0)
                    batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

                else:
                    batch_rois[index] = cur_roi[sampled_inds]
                    batch_roi_ious[index] = max_overlaps[sampled_inds]
                    batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

                batch_roi_scores[index] = cur_roi_scores[sampled_inds]

                if 'valid_length' in batch_dict.keys():
                    batch_valid_length[index] = cur_valid_length[:, sampled_inds]

            batch_rois_list.append(batch_rois)
            batch_gt_of_rois_list.append(batch_gt_of_rois)
            batch_roi_ious_list.append(batch_roi_ious)
            batch_roi_scores_list.append(batch_roi_scores)
            batch_roi_labels_list.append(batch_roi_labels)
        if self.roi_sampler_cfg.USE_TRAJ_AUG.ENABLED:
            batch_trajectory_rois_list = []
            for idx in range(0, batch_dict['num_frames']):
                if idx == cur_frame_idx:
                    batch_trajectory_rois_list.append(
                        cur_trajectory_rois[cur_frame_idx:cur_frame_idx + 1, sampled_inds])
                    continue
                fg_trajs, _ = self.aug_roi_by_noise_torch(cur_trajectory_rois[idx, fg_inds],
                                                          cur_trajectory_rois[idx, fg_inds][:, :8],
                                                          max_overlaps[fg_inds], \
                                                          aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES,
                                                          pos_thresh=self.roi_sampler_cfg.USE_TRAJ_AUG.THRESHOD)
                bg_trajs = cur_trajectory_rois[idx, bg_inds]
                batch_trajectory_rois_list.append(torch.cat([fg_trajs, bg_trajs], 0)[None, :, :])
            # batch_trajectory_rois[index] = torch.cat(batch_trajectory_rois_list, 0)
        else:
            pass
            # batch_trajectory_rois[index] = cur_trajectory_rois[:, sampled_inds]

        return torch.concat(batch_rois_list, 1), torch.concat(batch_gt_of_rois_list, 1), torch.concat(
            batch_roi_ious_list,
            1), torch.concat(batch_roi_scores_list, 1), torch.concat(batch_roi_labels_list,
                                                                     1)

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                        (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = torch.tensor([]).type_as(fg_inds)

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds.long(), fg_inds.long(), bg_inds.long()

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10, pos_thresh=None):
        iou_of_rois = torch.zeros(roi_boxes3d.shape[0]).type_as(gt_boxes3d)
        if pos_thresh is None:
            pos_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]

            gt_box3d = gt_boxes3d[k].view(1, gt_boxes3d.shape[-1])
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() <= self.roi_sampler_cfg.RATIO:
                    aug_box3d = roi_box3d  # p=RATIO to keep the original roi box
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = aug_box3d.view((1, aug_box3d.shape[-1]))
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(aug_box3d.cuda()[:, :7], gt_box3d.cuda()[:, :7]).to(device)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def random_aug_box3d(self, box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """

        if self.roi_sampler_cfg.REG_AUG_METHOD == 'single':
            pos_shift = (torch.rand(3, device=box3d.device) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.01) + 1.0  #
            angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]
            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot, box3d[7:]],
                                  dim=0)
            return aug_box3d
        elif self.roi_sampler_cfg.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            idx = torch.randint(low=0, high=len(range_config), size=(1,))[0].long()

            pos_shift = ((torch.rand(3, device=box3d.device) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((torch.rand(3, device=box3d.device) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((torch.rand(1, device=box3d.device) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
            return aug_box3d
        elif self.roi_sampler_cfg.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((torch.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift], dtype=np.float32)
            aug_box3d = torch.from_numpy(aug_box3d).type_as(box3d)
            return aug_box3d
        else:
            raise NotImplementedError


class Cross_attention_layer(nn.Module):
    def __init__(self, d_model, dropout=0.1, dim_feedforward=2048, activation='relu'):
        super().__init__()
        self.linear_attn_v = nn.Linear(d_model, d_model)
        self.linear_attn_out = nn.Linear(d_model, d_model)

        def get_activation_fn(activation):
            if activation == 'relu':
                return F.relu

        self.activation = get_activation_fn(activation)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, atten_map):
        if len(query.shape) == 4:
            batch_size = query.shape[0]
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            atten_map = atten_map.reshape(-1, atten_map.shape[-2], atten_map.shape[-1])
        else:
            batch_size = None

        value = self.linear_attn_v(key)
        pre_feature = torch.bmm(atten_map, value)
        pre_feature = self.linear_attn_out(pre_feature)

        query = query + self.dropout1(pre_feature)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm2(query)
        if batch_size == None:
            return query
        else:
            return query.reshape(batch_size, -1, query.shape[-2], query.shape[-1])


class VelocityHead(RoIHeadTemplate):
    def __init__(self, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.proposal_target_layer = ProposalTargetLayerMPPNet(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.use_time_stamp = self.model_cfg.get('USE_TIMESTAMP', None)
        self.num_lidar_points = self.model_cfg.Transformer.num_lidar_points
        self.avg_stage1_score = self.model_cfg.get('AVG_STAGE1_SCORE', None)

        self.nhead = model_cfg.Transformer.nheads
        self.num_enc_layer = model_cfg.Transformer.enc_layers
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.num_groups = model_cfg.Transformer.num_groups
        self.voxel_sampler = None
        self.grid_size = model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.num_proxy_points = model_cfg.Transformer.num_proxy_points
        self.seqboxembed = PointNet2(10, model_cfg=self.model_cfg)
        self.jointembed = MLP(self.hidden_dim * (self.num_groups + 1), model_cfg.Transformer.hidden_dim,
                              self.box_coder.code_size * self.num_class, 4)

        num_radius = len(self.model_cfg.ROI_GRID_POOL.POOL_RADIUS)
        self.up_dimension_geometry = MLP(input_dim=29, hidden_dim=256, output_dim=hidden_dim, num_layers=3)
        self.up_dimension_motion = MLP(input_dim=30, hidden_dim=64, output_dim=hidden_dim, num_layers=3)
        self.box_cls = MLP(input_dim=384, output_dim=1, hidden_dim=256, num_layers=3)
        self.box_reg = MLP(input_dim=384, output_dim=7, hidden_dim=256, num_layers=3)

        self.transformer = build_transformer(model_cfg.Transformer)

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=self.model_cfg.ROI_GRID_POOL.MLPS,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        self.cross_atten_decode_layer = nn.ModuleList()
        for i in range(3):
            self.cross_atten_decode_layer.append(Cross_attention_layer(d_model=256))

        self.class_embed = nn.ModuleList()
        for _ in range(self.num_enc_layer):
            self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim, 1))

        self.bbox_embed = nn.ModuleList()
        for _ in range(self.num_enc_layer):
            self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim,
                                       self.box_coder.code_size * self.num_class, 4))

        if self.model_cfg.Transformer.use_grid_pos.enabled:
            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                self.grid_index = torch.cat([i.reshape(-1, 1) for i in
                                             torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size),
                                                            torch.arange(self.grid_size))], 1).float().cuda()
                self.grid_pos_embeded = MLP(input_dim=4, hidden_dim=128, output_dim=hidden_dim, num_layers=2)
            else:
                self.pos = nn.Parameter(torch.zeros(1, self.num_grid_points, 256))

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.bbox_embed.layers[-1].weight, mean=0, std=0.001)

    def get_corner_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)
        local_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()

        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

    def roi_grid_pool(self, batch_size, rois, point_coords, point_features, batch_dict=None, batch_cnt=None):

        num_frames = batch_dict['num_frames']
        num_rois = rois.shape[2] * rois.shape[1]

        global_roi_proxy_points, local_roi_proxy_points = self.get_proxy_points_of_roi(
            rois.permute(0, 2, 1, 3).contiguous(), grid_size=self.grid_size
        )

        global_roi_proxy_points = global_roi_proxy_points.view(batch_size, -1, 3)

        point_coords = point_coords.view(point_coords.shape[0] * num_frames, point_coords.shape[1] // num_frames,
                                         point_coords.shape[-1])
        xyz = point_coords[:, :, 0:3].view(-1, 3)

        num_points = point_coords.shape[1]
        num_proxy_points = self.num_proxy_points

        if batch_cnt is None:
            xyz_batch_cnt = torch.tensor([num_points] * num_rois * batch_size).cuda().int()
        else:
            xyz_batch_cnt = torch.tensor(batch_cnt).cuda().int()

        new_xyz_batch_cnt = torch.tensor([num_proxy_points] * num_rois * batch_size).cuda().int()
        new_xyz = global_roi_proxy_points.view(-1, 3)

        _, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.view(-1, point_features.shape[-1]).contiguous(),
        )

        features = pooled_features.view(
            point_features.shape[0], num_frames * self.num_proxy_points,
            pooled_features.shape[-1]).contiguous()

        return features, global_roi_proxy_points.view(batch_size * rois.shape[2], num_frames * num_proxy_points,
                                                      3).contiguous()

    def get_proxy_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)
        local_roi_grid_points = common_utils.rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(
            dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0, 3, 6, 9, 12, 15, 18, 21, 24]).to(device)  #
        indices_y = torch.LongTensor([1, 4, 7, 10, 13, 16, 19, 22, 25]).to(device)  #
        indices_z = torch.LongTensor([2, 5, 8, 11, 14, 17, 20, 23, 26]).to(device)
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim=-1)
        return src

    def crop_current_frame_points(self, src, batch_size, trajectory_rois, num_rois, batch_dict):

        for bs_idx in range(batch_size):
            cur_batch_boxes = trajectory_rois[bs_idx, 0, :, :7].view(-1, 7)
            cur_radiis = torch.sqrt((cur_batch_boxes[:, 3] / 2) ** 2 + (cur_batch_boxes[:, 4] / 2) ** 2) * 1.1
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]
            cur_points = cur_points[cur_points[:, -1] == 0]
            dis = torch.norm((cur_points[:, :2].unsqueeze(0) - cur_batch_boxes[:, :2].unsqueeze(1).repeat(1,
                                                                                                          cur_points.shape[
                                                                                                              0], 1)),
                             dim=2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))

            sampled_idx = torch.topk(point_mask.float(), 128)[1]
            sampled_idx_buffer = sampled_idx[:, 0:1].repeat(1, 128)
            roi_idx = torch.arange(num_rois)[:, None].repeat(1, 128)
            sampled_mask = point_mask[roi_idx, sampled_idx]
            sampled_idx_buffer[sampled_mask] = sampled_idx[sampled_mask]

            src[bs_idx] = cur_points[sampled_idx_buffer][:, :, :5]
            empty_flag = sampled_mask.sum(-1) == 0
            src[bs_idx, empty_flag] = 0

        src = src.repeat([1, 1, trajectory_rois.shape[1], 1])

        return src

    def crop_previous_frame_points(self, src, batch_size, trajectory_rois, num_rois, valid_length, batch_dict):
        for bs_idx in range(batch_size):

            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]

            for idx in range(1, trajectory_rois.shape[1]):

                time_mask = (cur_points[:, -1] - idx * 0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask]
                cur_batch_boxes = trajectory_rois[bs_idx, idx, :, :7].view(-1, 7)

                cur_radiis = torch.sqrt((cur_batch_boxes[:, 3] / 2) ** 2 + (cur_batch_boxes[:, 4] / 2) ** 2) * 1.1
                if not self.training and cur_batch_boxes.shape[0] > 32:
                    length_iter = cur_batch_boxes.shape[0] // 32
                    dis_list = []
                    for i in range(length_iter + 1):
                        dis = torch.norm((cur_time_points[:, :2].unsqueeze(0) - \
                                          cur_batch_boxes[32 * i:32 * (i + 1), :2].unsqueeze(1).repeat(1,
                                                                                                       cur_time_points.shape[
                                                                                                           0], 1)),
                                         dim=2)
                        dis_list.append(dis)
                    dis = torch.cat(dis_list, 0)
                else:
                    dis = torch.norm((cur_time_points[:, :2].unsqueeze(0) - \
                                      cur_batch_boxes[:, :2].unsqueeze(1).repeat(1, cur_time_points.shape[0], 1)),
                                     dim=2)

                point_mask = (dis <= cur_radiis.unsqueeze(-1)).view(trajectory_rois.shape[2], -1)

                for roi_box_idx in range(0, num_rois):

                    if not valid_length[bs_idx, idx, roi_box_idx]:
                        continue

                    cur_roi_points = cur_time_points[point_mask[roi_box_idx]]

                    if cur_roi_points.shape[0] > self.num_lidar_points:
                        np.random.seed(0)
                        choice = np.random.choice(cur_roi_points.shape[0], self.num_lidar_points, replace=True)
                        cur_roi_points_sample = cur_roi_points[choice]

                    elif cur_roi_points.shape[0] == 0:
                        cur_roi_points_sample = cur_roi_points.new_zeros(self.num_lidar_points, 6)

                    else:
                        empty_num = self.num_lidar_points - cur_roi_points.shape[0]
                        add_zeros = cur_roi_points.new_zeros(empty_num, 6)
                        add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                        cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim=0)

                    if not self.use_time_stamp:
                        cur_roi_points_sample = cur_roi_points_sample[:, :-1]

                    src[bs_idx, roi_box_idx, self.num_lidar_points * idx:self.num_lidar_points * (idx + 1),
                    :] = cur_roi_points_sample

        return src

    def get_proposal_aware_geometry_feature(self, src, batch_size, trajectory_rois, num_rois,t=0):
        proposal_aware_feat_list = []


        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois)

        # corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
        corner_points = corner_points.view(batch_size * num_rois, -1)
        trajectory_roi_center = trajectory_rois[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
        proposal_aware_feat = src[:, :, :3].repeat(1, 1,9) - \
                              corner_add_center_points.unsqueeze(1).repeat(1, src.shape[1], 1)

        lwh = trajectory_rois[:,  :][:, 3:6].unsqueeze(1)
        diag_dist = ((lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5).repeat(1,proposal_aware_feat.shape[1])
        proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))
        pos = (src[:,:,:3] - trajectory_roi_center[:,None,:])/diag_dist.unsqueeze(-1)
        pos = common_utils.rotate_points_along_z(points=pos,angle=-trajectory_rois[:,6])
        pos = torch.concat([pos,torch.full((pos.shape[0],pos.shape[1],1),t).to(device)],dim=-1)

        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:, :, 3:]], dim=-1)
        src_gemoetry = self.up_dimension_geometry(proposal_aware_feat)

        return src_gemoetry,pos

    def get_proposal_aware_motion_feature(self, proxy_point, batch_size, trajectory_rois, num_rois, batch_dict):

        time_stamp = torch.ones([proxy_point.shape[0], proxy_point.shape[1], 1]).to(device)
        padding_zero = torch.zeros([proxy_point.shape[0], proxy_point.shape[1], 2]).to(device)
        proxy_point_time_padding = torch.cat([padding_zero, time_stamp], -1)

        num_frames = trajectory_rois.shape[1]

        for i in range(num_frames):
            proxy_point_time_padding[:, i * self.num_proxy_points:(i + 1) * self.num_proxy_points, -1] = i * 0.1

        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:, 0, :, :].contiguous())
        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
        corner_points = corner_points.view(batch_size * num_rois, -1)
        trajectory_roi_center = trajectory_rois[:, 0, :, :].reshape(batch_size * num_rois, -1)[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)

        proposal_aware_feat = proxy_point[:, :, :3].repeat(1, 1, 9) - corner_add_center_points.unsqueeze(1)

        lwh = trajectory_rois[:, 0, :, :].reshape(batch_size * num_rois, -1)[:, 3:6].unsqueeze(1).repeat(1,
                                                                                                         proxy_point.shape[
                                                                                                             1], 1)
        diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
        proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))

        proposal_aware_feat = torch.cat([proposal_aware_feat, proxy_point_time_padding], -1)
        proxy_point_motion_feat = self.up_dimension_motion(proposal_aware_feat)

        return proxy_point_motion_feat

    def trajectories_auxiliary_branch(self, trajectory_rois):

        time_stamp = torch.ones([trajectory_rois.shape[0], trajectory_rois.shape[1], trajectory_rois.shape[2], 1]).to(
            device)
        for i in range(time_stamp.shape[1]):
            time_stamp[:, i, :] = i * 0.1

        box_seq = torch.cat([trajectory_rois[:, :, :, :], time_stamp], -1)

        box_seq[:, :, :, 0:3] = box_seq[:, :, :, 0:3] - box_seq[:, 0:1, :, 0:3]

        roi_ry = box_seq[:, :, :, 6] % (2 * np.pi)
        roi_ry_t0 = roi_ry[:, 0]
        roi_ry_t0 = roi_ry_t0.repeat(1, box_seq.shape[1])

        box_seq = common_utils.rotate_points_along_z(
            points=box_seq.view(-1, 1, box_seq.shape[-1]), angle=-roi_ry_t0.view(-1)
        ).view(box_seq.shape[0], box_seq.shape[1], -1, box_seq.shape[-1])

        box_seq[:, :, :, 6] = 0

        batch_rcnn = box_seq.shape[0] * box_seq.shape[2]
        box_seq[:, :, :, :2] -= box_seq[:, :, :, 7:9] * time_stamp * 10

        box_reg, box_feat = self.seqboxembed(
            box_seq.contiguous())

        return box_reg, box_feat

    def attention_weight_map_cal(self, pre_boxes, cur_boxes, scores_pre, labels_pre, labels_cur):

        B = pre_boxes.shape[0]
        iou3d_list = []
        for b in range(B):
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_boxes[b][:, :7], pre_boxes[b][:, :7])

            # velocity_diff = ((cur_boxes[b][:, -2:][:, None, :] - cur_boxes[b][:, -2:][None, :, :]) ** 2).sum(dim=-1)
            iou3d_list.append(iou3d[None, :, :])
        iou3d = torch.concat(iou3d_list, dim=0)
        iou3d *= scores_pre[:, None, :]
        labels_mask = labels_cur[:, :, None] == labels_pre[:, None, :]
        iou3d *= labels_mask

        attention = iou3d / torch.clamp(iou3d.sum(dim=-1), min=1)[:, :, None]

        return attention

    def generate_trajectory(self, cur_batch_boxes, proposals_list, batch_dict):

        trajectory_rois = cur_batch_boxes[:, None, :, :].repeat(1, batch_dict['rois'].shape[-2], 1, 1)
        trajectory_rois[:, 0, :, :] = cur_batch_boxes
        valid_length = torch.zeros([batch_dict['batch_size'], batch_dict['rois'].shape[-2], trajectory_rois.shape[2]])
        valid_length[:, 0] = 1
        num_frames = batch_dict['rois'].shape[-2]
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = trajectory_rois[:, i - 1, :, 0:2] + trajectory_rois[:, i - 1, :, 7:9]
            frame[:, :, 2:] = trajectory_rois[:, i - 1, :, 2:]

            for bs_idx in range(batch_dict['batch_size']):
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame.cuda()[bs_idx, :, :7],
                                                        proposals_list.cuda()[bs_idx, i, :, :7]).to(device)

                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)

                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)

                valid_length[bs_idx, i, fg_inds] = 1

                trajectory_rois[bs_idx, i, fg_inds, :] = proposals_list[bs_idx, i, traj_assignment[fg_inds]]

            batch_dict['valid_length'] = valid_length

        return trajectory_rois, valid_length

    def exchange_box(self, attention_map, boxes_pre, boxes_cur, scores_pre, scores_cur, cur_features, pre_features,
                     return_idx=False):
        assert boxes_pre.shape[1] == scores_pre.shape[1]
        assert boxes_cur.shape[1] == scores_cur.shape[1]
        batch_size = attention_map.shape[0]
        max_exchanged_boxes = 5
        pre_location_mask = (boxes_pre[:, :, :2].abs() > 70).sum(dim=-1) == 0
        cur_location_mask = (boxes_cur[:, :, :2].abs() > 70).sum(dim=-1) == 0

        # pre_mask = attention_map.sum(dim=-1)<0.1
        pre_mask = attention_map.sum(dim=-1) < 0.2
        pre_mask *= pre_location_mask
        cur_mask = attention_map.sum(dim=-2) < 0.2
        cur_mask *= cur_location_mask
        index_cur_all = torch.full((batch_size, max_exchanged_boxes), -1).to(device)
        for bs in range(attention_map.shape[0]):
            score_batch_pre, ind_batch_pre = torch.topk(scores_pre[bs][pre_mask[bs]],
                                                        k=min(max_exchanged_boxes, pre_mask[bs].sum()), dim=-1,
                                                        largest=True)
            score_batch_cur, ind_batch_cur = torch.topk(scores_cur[bs][cur_mask[bs]],
                                                        k=min(max_exchanged_boxes, cur_mask[bs].sum()), dim=-1,
                                                        largest=False)
            num_exchanged_batch = min((score_batch_pre > 0.7).sum(), (score_batch_cur < 0.3).sum())
            index_pre = torch.where(scores_pre[bs][:, None] == score_batch_pre[:num_exchanged_batch][None, :])[0][
                        :num_exchanged_batch]
            index_cur = torch.where(scores_cur[bs][:, None] == score_batch_cur[:num_exchanged_batch][None, :])[0][
                        :num_exchanged_batch]
            boxes_cur[bs][index_cur] = boxes_pre[bs][index_pre]
            cur_features[bs][index_cur] = pre_features[bs][index_pre]

            scores_cur[bs][index_cur] = scores_pre[bs][index_pre]
            index_cur_all[bs][:num_exchanged_batch] = index_cur
            attention_map[bs, :, index_cur] = 0
            attention_map[bs, :, index_cur][index_pre] = 1

        if not return_idx:
            return boxes_cur, cur_features, scores_cur, attention_map
        else:
            return boxes_cur, cur_features, scores_cur, index_cur_all, attention_map

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        rois = batch_dict['proposals_list']
        # roi_scores = batch_dict['roi_scores']
        # roi_labels = batch_dict['roi_labels']
        # rois = rois.reshape(-1,rois.shape[-2],rois.shape[-1])
        # roi_scores = roi_scores.reshape(-1,roi_scores.shape[-1])
        # roi_labels = roi_labels.reshape(-1,roi_labels.shape[-1])
        # selected = [nms_gpu(rois[i,:,:7],roi_scores[i],thresh=0.7,pre_maxsize=(roi_scores[i]>0).sum())[0] for i in range(rois.shape[0])]
        # rois_list = [rois[i][selected[i]] for i in range(rois.shape[0])]
        # roi_scores_list = [roi_scores[i][selected[i]] for i in range(roi_scores.shape[0])]
        # roi_labels_list = [roi_labels[i][selected[i]] for i in range(roi_labels.shape[0])]
        #
        # max_roi_num = max([roi.shape[0] for roi in rois_list])
        # rois = rois.new_zeros(rois.shape[0],max_roi_num,rois.shape[-1])
        # roi_scores = roi_scores.new_zeros(roi_scores.shape[0],max_roi_num)
        # roi_labels = roi_labels.new_zeros(roi_labels.shape[0],max_roi_num)
        # for i in range(rois.shape[0]):
        #     rois[i,:rois_list[i].shape[0]] = rois_list[i]
        #     roi_labels[i,:roi_labels_list[i].shape[0]] = roi_labels_list[i]
        #     roi_scores[i,:roi_scores_list[i].shape[0]] = roi_scores_list[i]
        #
        # rois = rois.reshape(batch_dict['batch_size'],-1,rois.shape[-2],rois.shape[-1])
        # roi_labels = roi_labels.reshape(batch_dict['batch_size'],-1,roi_labels.shape[-1])
        # roi_scores = roi_scores.reshape(batch_dict['batch_size'],-1,roi_scores.shape[-1])
        # batch_dict['rois_nms'] = rois
        # batch_dict['roi_labels_nms'] = roi_labels.long()
        # batch_dict['roi_scores_nms'] = roi_scores
        
        # batch_dict['rois'] = batch_dict['proposals_list'].permute(0, 2, 1, 3)
        num_rois = batch_dict['roi_boxes'].shape[1]
        batch_dict['num_frames'] = batch_dict['roi_boxes'].shape[1]
        # batch_dict['roi_scores'] = batch_dict['roi_scores'].permute(0, 2, 1)
        batch_dict['roi_labels'] = batch_dict['roi_labels'].long()
        proposals_list = batch_dict['proposals_list']
        batch_size = batch_dict['batch_size']
        cur_batch_boxes = copy.deepcopy(batch_dict['roi_boxes'].detach())[:, :, 0]
        batch_dict['cur_frame_idx'] = 0

        # trajectory_rois, valid_length = self.generate_trajectory(cur_batch_boxes, proposals_list, batch_dict)

        # batch_dict['traj_memory'] = trajectory_rois
        batch_dict['has_class_labels'] = True
        # batch_dict['trajectory_rois'] = trajectory_rois
        # rois = batch_dict['rois']
        if self.voxel_sampler is None:
            self.voxel_sampler = build_voxel_sampler(device)

        rois = batch_dict['roi_boxes']
        rois = rois.reshape(batch_size, self.model_cfg.NUM_FRAMES, -1, rois.shape[-1])
        num_rois = batch_dict['roi_boxes'].shape[1] // batch_dict['num_frames']
        num_sample = self.num_lidar_points
        batch_dict = self.voxel_sampler(batch_size, rois, num_sample[0],
                                 batch_dict)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['trajectory_rois']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # targets_dict['trajectory_rois'][:, :, :, :] = batch_dict['rois'].reshape(
            #     batch_size, batch_dict['num_frames'], -1, batch_dict['rois'].shape[-1])
            # trajectory_rois = targets_dict['trajectory_rois']
            # valid_length = targets_dict['valid_length']
            empty_mask = batch_dict['rois'][:, :, :6].sum(-1) == 0
        else:
            empty_mask = batch_dict['rois'][:, :, 0, :6].sum(-1) == 0
            batch_dict['valid_traj_mask'] = ~empty_mask


        # src = rois.new_zeros(batch_size, num_rois, num_sample, 5)


        rois_batch = rois
        num_frames = self.model_cfg.NUM_FRAMES
        for f in range(self.model_cfg.NUM_FRAMES):
            # src = self.crop_current_frame_points(src, batch_size, trajectory_rois, num_rois, batch_dict)

            # src = self.crop_previous_frame_points(src, batch_size, trajectory_rois, num_rois, valid_length, batch_dict)

            src_geometry_feature ,pos= self.get_proposal_aware_geometry_feature(src.reshape(-1,self.num_lidar_points[f],src.shape[-1]), batch_size, rois_batch.reshape(-1,rois_batch.shape[-1]),
                                                                                          num_rois*num_frames)
            # src_motion_feature = self.get_proposal_aware_motion_feature(src, batch_size, rois_batch, num_rois)

            # src_motion_feature = self.get_proposal_aware_motion_feature(proxy_points, batch_size, trajectory_rois, num_rois,
            #                                                             batch_dict)
            src = src_geometry_feature
            if self.model_cfg.get('USE_TRAJ_EMPTY_MASK', None):
                src[empty_mask.view(-1)] = 0

            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                pos = self.grid_pos_embeded(pos)

            else:
                pos = None
            hs, tokens = self.transformer[f](src, pos=pos)
        point_cls_list = []
        point_reg_list = []

        for i in range(self.num_enc_layer):
            point_cls_list.append(self.class_embed[i](tokens[i][0]))

        for i in range(self.num_enc_layer):
            point_reg_list.append(self.bbox_embed[i](tokens[i][0]))

        point_cls = torch.cat(point_cls_list, 0).reshape(batch_size, self.num_enc_layer, -1, 1)

        point_reg = torch.cat(point_reg_list, 0).reshape(batch_size, self.num_enc_layer, -1, 7)

        # hs = hs.permute(1, 0, 2).reshape(hs.shape[1], -1)

        # joint_reg = self.jointembed(torch.cat([hs, feat_box], -1))
        features = hs.reshape(batch_size, -1, hs.shape[-2], hs.shape[-1])
        rois = rois.reshape(batch_size, self.model_cfg.NUM_FRAMES, -1, rois.shape[-1])
        features_list = []
        roi_scores, roi_labels = targets_dict['roi_scores'].reshape(batch_size, -1, num_rois), targets_dict[
            'roi_labels'].reshape(batch_size, -1, num_rois)

        for i in range(self.model_cfg.NUM_FRAMES - 1):
            cur_features = features[:, :-1]

            pre_features = features[:, 1:]
            boxes_pre = rois[:, 1:-i] if i > 0 else rois[:, 1:]
            boxes_pre[:, :, :, :2] -= boxes_pre[:, :, :, -2:]
            scores_pre = roi_scores[:, 1:] if i == 0 else roi_scores[:, 1:-i]
            labels_pre = roi_labels[:, 1:] if i == 0 else roi_labels[:, 1:-i]

            boxes_cur = rois[:, :-i - 1]
            scores_cur = roi_scores[:, -i - 1]
            labels_cur = roi_labels[:, :-i - 1]

            atten_weight_map = self.attention_weight_map_cal(
                boxes_pre.reshape(-1, boxes_pre.shape[-2], boxes_pre.shape[-1]),
                boxes_cur.reshape(-1, boxes_cur.shape[-2],
                                  boxes_cur.shape[-1]), scores_pre.reshape(-1, scores_pre.shape[-1]),
                labels_pre.reshape(-1, labels_pre.shape[-1]),
                labels_cur.reshape(-1, labels_cur.shape[-1])).reshape(
                batch_size, -1, boxes_pre.shape[-2], boxes_cur.shape[-2])
            if not self.training:
                if i < self.model_cfg.NUM_FRAMES - 2:
                    rois[:, :-i - 1], cur_features[:, -1], scores_cur, atten_weight_map[:, -1] = self.exchange_box(
                        atten_weight_map[:, -1].clone(), boxes_pre[:, -1], boxes_cur[:, -1],
                        roi_scores[:, -i - 1], roi_scores[:, -i], cur_features[:, -1],
                        pre_features[:, -1])
                else:
                    rois[:, :-i - 1], cur_features[:, -1], scores_cur, idx, atten_weight_map[:, -1] = self.exchange_box(
                        atten_weight_map[:, -1].clone(),
                        boxes_pre[:, -1].clone(),
                        boxes_cur[:, -1].clone(),
                        roi_scores[:, -i - 1],
                        roi_scores[:, -i],
                        cur_features[:, -1],
                        pre_features[:, -1], return_idx=True)

            features = self.cross_atten_decode_layer[i](pre_features, cur_features, atten_weight_map)

            features_list.append(features[:, 0])

        features = torch.concat(features_list, dim=1)
        box_reg, feat_box = self.trajectories_auxiliary_branch(rois)
        features = torch.concat([features, feat_box.repeat(1, self.model_cfg.NUM_FRAMES - 1, 1)], dim=-1)

        # rcnn_cls = point_cls
        # rcnn_reg = joint_reg
        if not self.training:
            idx_mask = torch.ones(batch_size, features.shape[-2], dtype=torch.bool)
            for b in range(batch_size):
                idx_mask[b][idx[b][idx[b] >= 0]] = False
        rcnn_cls = self.box_cls(features)
        rcnn_reg = self.box_reg(features)

        if not self.training:
            batch_dict['rois'] = batch_dict['rois'][:, :, 0].contiguous()
            rcnn_cls = rcnn_cls[-rcnn_cls.shape[0] // self.num_enc_layer:]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            batch_dict['cls_preds_normalized'] = False
            if self.avg_stage1_score:
                stage1_score = batch_dict['roi_scores'][:, :, :1]
                batch_cls_preds = F.sigmoid(batch_cls_preds)
                if self.model_cfg.get('IOU_WEIGHT', None):
                    batch_box_preds_list = []
                    roi_labels_list = []
                    batch_cls_preds_list = []
                    for bs_idx in range(batch_size):
                        car_mask = batch_dict['roi_labels'][bs_idx] == 1
                        batch_cls_preds_car = batch_cls_preds[bs_idx].pow(self.model_cfg.IOU_WEIGHT[0]) * \
                                              stage1_score[bs_idx].pow(1 - self.model_cfg.IOU_WEIGHT[0])
                        batch_cls_preds_car = batch_cls_preds_car[car_mask][None]
                        batch_cls_preds_pedcyc = batch_cls_preds[bs_idx].pow(self.model_cfg.IOU_WEIGHT[1]) * \
                                                 stage1_score[bs_idx].pow(1 - self.model_cfg.IOU_WEIGHT[1])
                        batch_cls_preds_pedcyc = batch_cls_preds_pedcyc[~car_mask][None]
                        cls_preds = torch.cat([batch_cls_preds_car, batch_cls_preds_pedcyc], 1)
                        box_preds = torch.cat([batch_dict['batch_box_preds'][bs_idx][car_mask],
                                               batch_dict['batch_box_preds'][bs_idx][~car_mask]], 0)[None]
                        roi_labels = torch.cat([batch_dict['roi_labels'][bs_idx][car_mask],
                                                batch_dict['roi_labels'][bs_idx][~car_mask]], 0)[None]
                        batch_box_preds_list.append(box_preds)
                        roi_labels_list.append(roi_labels)
                        batch_cls_preds_list.append(cls_preds)
                    batch_dict['batch_box_preds'] = torch.cat(batch_box_preds_list, 0)
                    batch_dict['roi_labels'] = torch.cat(roi_labels_list, 0)
                    batch_cls_preds = torch.cat(batch_cls_preds_list, 0)

                else:
                    batch_cls_preds = torch.sqrt(batch_cls_preds * stage1_score)
                batch_dict['cls_preds_normalized'] = True

            batch_dict['batch_cls_preds'] = batch_cls_preds


        else:
            targets_dict['batch_size'] = batch_size
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['box_reg'] = box_reg
            targets_dict['point_reg'] = point_reg
            targets_dict['point_cls'] = point_cls
            if not self.training:
                targets_dict['idx'] = idx
            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        batch_size = forward_ret_dict['batch_size']

        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)

        rcnn_reg = forward_ret_dict['rcnn_reg']

        roi_boxes3d = forward_ret_dict['rois']

        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':

            rois_anchor = roi_boxes3d.clone().detach()[:, :, :7].contiguous().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            num_groups = rcnn_reg.shape[1] // gt_boxes3d_ct.shape[1]
            num_features = gt_boxes3d_ct.shape[1]
            # gt_boxes3d_ct = gt_boxes3d_ct.repeat(1,num_groups,1)
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )
            rcnn_loss_reg = 0
            for i in range(num_groups):
                rcnn_loss_reg_ = self.reg_loss_func(
                    rcnn_reg[:, num_features * i:num_features * (i + 1)].view(rcnn_batch_size, -1).unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0),
                )  # [B, M, 7]
                rcnn_loss_reg += (rcnn_loss_reg_.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(
                    dim=-1).float()).sum() / max(
                    fg_sum, 1)
            rcnn_loss_reg /= num_groups

            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                            loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if self.model_cfg.USE_AUX_LOSS:
                point_reg = forward_ret_dict['point_reg']

                groups = point_reg.shape[1]
                if groups != 1:
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                            point_reg[:, i].view(slice, -1).unsqueeze(dim=0),
                            reg_targets.unsqueeze(dim=0), )
                        point_loss_reg = (point_loss_reg.view(slice, -1) * fg_mask.unsqueeze(
                            dim=-1).float()).sum() / max(fg_sum, 1)
                        point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                                         loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]

                        point_loss_regs += point_loss_reg
                    point_loss_regs = point_loss_regs / groups
                    tb_dict['point_loss_reg'] = point_loss_regs.item()
                    rcnn_loss_reg += point_loss_regs

                else:
                    point_loss_reg = self.reg_loss_func(point_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                                                        reg_targets.unsqueeze(dim=0), )
                    point_loss_reg = (point_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(
                        dim=-1).float()).sum() / max(fg_sum, 1)
                    point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                                     loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]
                    tb_dict['point_loss_reg'] = point_loss_reg.item()
                    rcnn_loss_reg += point_loss_reg

                seqbox_reg = forward_ret_dict['box_reg']
                seqbox_loss_reg = self.reg_loss_func(seqbox_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                                                     reg_targets.unsqueeze(dim=0), )
                seqbox_loss_reg = (seqbox_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(
                    dim=-1).float()).sum() / max(fg_sum, 1)
                seqbox_loss_reg = seqbox_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                                  loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][1]
                tb_dict['seqbox_loss_reg'] = seqbox_loss_reg.item()
                rcnn_loss_reg += seqbox_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                for i in range(num_groups):
                    fg_rcnn_reg = rcnn_reg[:, i * num_features:(i + 1) * num_features].view(rcnn_batch_size, -1)[
                        fg_mask]
                    fg_roi_boxes3d = roi_boxes3d[:, :, :7].contiguous().view(-1, code_size)[fg_mask]

                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.box_coder.decode_torch(
                        fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                    ).view(-1, code_size)

                    rcnn_boxes3d = common_utils.rotate_points_along_z(
                        rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                    ).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz

                    corner_loss_func = loss_utils.get_corner_loss_lidar

                    loss_corner = corner_loss_func(
                        rcnn_boxes3d[:, 0:7],
                        gt_of_rois_src[fg_mask][:, 0:7])

                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner.item()

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':

            rcnn_cls_flat = rcnn_cls.view(-1)

            groups = rcnn_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
            if groups != 1:
                rcnn_loss_cls = 0
                slice = rcnn_cls_labels.shape[0]
                for i in range(groups):
                    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat[i * slice:(i + 1) * slice]),
                                                            rcnn_cls_labels.float(), reduction='none')

                    cls_valid_mask = (rcnn_cls_labels >= 0).float()
                    rcnn_loss_cls = rcnn_loss_cls + (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(
                        cls_valid_mask.sum(), min=1.0)

                rcnn_loss_cls = rcnn_loss_cls / groups

            else:

                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(),
                                                        reduction='none')
                cls_valid_mask = (rcnn_cls_labels >= 0).float()
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)


        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']

        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds=None, box_preds=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)
        Returns:
        """
        code_size = self.box_coder.code_size
        if cls_preds is not None:
            batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        else:
            batch_cls_preds = None
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds, rois[:, :, 7:]], -1)
        return batch_cls_preds, batch_box_preds
