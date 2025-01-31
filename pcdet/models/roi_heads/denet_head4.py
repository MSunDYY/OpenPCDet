from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.functional as F
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.denet_utils import build_transformer, PointNet, MLP, SpatialMixerBlock, build_voxel_sampler_denet
from ..model_utils.msf_utils import build_voxel_sampler_traj, build_voxel_sampler_anchor,build_voxel_sampler
from . import msf_head
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet import device


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

        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_labels, \
            batch_backward_rois, batch_valid_length = self.sample_rois_for_mppnet(batch_dict=batch_dict)

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

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois,
                        'gt_iou_of_rois': batch_roi_ious,  # 'roi_scores': batch_roi_scores,
                        'roi_labels': batch_roi_labels, 'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'trajectory_rois': batch_backward_rois,
                        'valid_length': batch_valid_length,
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
        cur_frame_idx = 0
        batch_size = batch_dict['batch_size']
        rois = batch_dict['trajectory_rois'][:, cur_frame_idx, :, :]
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']
        num_anchors = batch_dict['num_anchors']
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors,
                                          gt_boxes.shape[-1])
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors),
                                          dtype=torch.long)
        backward_rois = batch_dict['trajectory_rois']
        # trajectory_rois = batch_dict['trajectory_rois']
        # batch_trajectory_rois = rois.new_zeros(batch_size, trajectory_rois.shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE,
        #                                         trajectory_rois.shape[-1])
        batch_backward_rois = rois.new_zeros(batch_size, backward_rois.shape[1],
                                             self.roi_sampler_cfg.ROI_PER_IMAGE * num_anchors,
                                             backward_rois.shape[-1])

        # valid_length = batch_dict['valid_length']
        batch_valid_length = rois.new_zeros(
            (batch_size, batch_dict['trajectory_rois'].shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE))

        for index in range(batch_size):

            cur_backward_rois = backward_rois[index]
            # cur_trajectory_rois = trajectory_rois[index]
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = rois[index], gt_boxes[index], rois[index][:, -1].long(), \
                roi_scores[index]

            if 'valid_length' in batch_dict.keys():
                cur_valid_length = batch_dict['valid_length'][index].to(device)

            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1

            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi[:, :-1], roi_labels=cur_roi[:, -1].long(),
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )

            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            num_rois = max_overlaps.shape[0] // num_anchors
            sampled_inds, fg_inds, bg_inds = self.subsample_rois(
                max_overlaps=max_overlaps[torch.arange(num_rois) * num_anchors])
            fg_inds = torch.concat([fg_inds[:, None] * num_anchors + i for i in range(num_anchors)], dim=-1).flatten()
            bg_inds = torch.concat([bg_inds[:, None] * num_anchors + i for i in range(num_anchors)], dim=-1).flatten()
            sampled_inds = torch.concat([fg_inds, bg_inds], dim=0)
            batch_roi_labels[index] = cur_roi_labels[sampled_inds.long()]

            if self.roi_sampler_cfg.get('USE_ROI_AUG', False):

                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(cur_roi[fg_inds], cur_gt[gt_assignment[fg_inds]],
                                                                max_overlaps[fg_inds],
                                                                aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES)
                if self.roi_sampler_cfg.get('USE_BG_ROI_AUG', False):
                    bg_rois, _ = self.aug_roi_by_noise_torch(cur_roi[bg_inds], cur_roi[bg_inds], max_overlaps[bg_inds],
                                                             aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES)
                    bg_iou3d = iou3d_nms_utils.boxes_iou3d_gpu(bg_rois[:, :7], cur_gt[:, :7][gt_assignment[bg_inds]])
                    bg_iou3d = bg_iou3d[torch.arange(bg_rois.shape[0]), torch.arange(bg_rois.shape[0])]
                else:
                    bg_rois = cur_roi[bg_inds]
                    bg_iou3d = max_overlaps[bg_inds]

                batch_rois[index] = torch.cat([fg_rois, bg_rois], 0)
                batch_roi_ious[index] = torch.cat([fg_iou3d, bg_iou3d], 0)
                batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

            else:
                batch_rois[index] = cur_roi[sampled_inds]
                batch_roi_ious[index] = max_overlaps[sampled_inds]
                batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

            # batch_roi_scores[index] = cur_roi_scores[sampled_inds]

            # if 'valid_length' in batch_dict.keys():
            #     batch_valid_length[index] = cur_valid_length[:, sampled_inds]

            if self.roi_sampler_cfg.USE_TRAJ_AUG.ENABLED:
                batch_backward_rois_list = []
                batch_trajectory_rois_list = []
                for idx in range(0, batch_dict['num_frames']):
                    if idx == cur_frame_idx:
                        batch_backward_rois_list.append(
                            cur_backward_rois[cur_frame_idx:cur_frame_idx + 1, sampled_inds])
                        # batch_trajectory_rois_list.append(
                        #     cur_trajectory_rois[cur_frame_idx:cur_frame_idx+1,sampled_inds]
                        # )
                        continue
                    fg_backs, _ = self.aug_roi_by_noise_torch(cur_backward_rois[idx, fg_inds],
                                                              cur_backward_rois[idx, fg_inds][:, :8],
                                                              max_overlaps[fg_inds], \
                                                              aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES,
                                                              pos_thresh=self.roi_sampler_cfg.USE_TRAJ_AUG.THRESHOD)
                    bg_backs = cur_backward_rois[idx, bg_inds]
                    # fg_trajs,_ = self.aug_roi_by_noise_torch(cur_trajectory_rois[idx,fg_inds],
                    #                                          cur_trajectory_rois[idx,fg_inds][:,:8],
                    #                                          max_overlaps[fg_inds],
                    #                                          aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES,
                    #                                          pos_thresh=self.roi_sampler_cfg.USE_TRAJ_AUG.THRESHOD
                    #                                          )
                    # bg_trajs = cur_trajectory_rois[idx,bg_inds]

                    batch_backward_rois_list.append(torch.cat([fg_backs, bg_backs], 0)[None, :, :])
                    # batch_trajectory_rois_list.append(torch.cat([fg_trajs,bg_trajs],0)[None,:,:])
                batch_backward_rois[index] = torch.cat(batch_backward_rois_list, 0)
                # batch_trajectory_rois[index] = torch.cat(batch_trajectory_rois_list,0)
            else:
                batch_backward_rois[index] = cur_backward_rois[:, sampled_inds]
                # batch_trajectory_rois[index] = cur_trajectory_rois[:,sampled_inds]
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_labels, batch_backward_rois, batch_valid_length

    def sample_rois_for_mppnet_(self, max_overlaps):
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
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(aug_box3d[:, :7], gt_box3d[:, :7])
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
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.15) + 1.0  #
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


class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, grid_size, channels, config=None, dropout=0.0):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.pos_linear = nn.Linear(3, channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
        self.config = config
        self.grid_size = grid_size

    def forward(self, src1, src2, xyz1, xyz2):
        xyz1 = self.pos_linear(xyz1).permute(1, 0, 2)
        xyz2 = self.pos_linear(xyz2).permute(1, 0, 2)
        src1 = src1.permute(1, 0, 2)
        src2 = src2.permute(1, 0, 2)
        src = self.mixer(src1 + xyz1, src2 + xyz2, src2)[0]

        src = src1 + self.dropout(src)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)

        return src_mixer.permute(1, 0, 2)


class DENet4Head(RoIHeadTemplate):
    def __init__(self, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.proposal_target_layer = ProposalTargetLayerMPPNet(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        # self.proposal_target_layer = msf_head.ProposalTargetLayerMPPNet(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.use_time_stamp = self.model_cfg.get('USE_TIMESTAMP', None)
        self.num_lidar_points = self.model_cfg.Transformer.num_lidar_points
        self.avg_stage1_score = self.model_cfg.get('AVG_STAGE1_SCORE', None)
        self.nhead = model_cfg.Transformer.nheads
        self.num_enc_layer = model_cfg.Transformer.enc_layers
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.num_anchors = model_cfg.NUM_ANCHORS
        self.num_groups = model_cfg.Transformer.num_groups

        self.grid_size = model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.hidden_dim),
            # nn.ReLU(inplace=False),
        )
        self.cross = nn.ModuleList([CrossAttention(3, 4, 256, None) for i in range(4)])
        self.seqboxembed = PointNet(8, model_cfg=self.model_cfg)
        self.jointembed = MLP((model_cfg.Transformer.hidden_dim+model_cfg.Transformer.point_dim) * self.num_groups+model_cfg.Transformer.hidden_dim,
                              model_cfg.Transformer.hidden_dim+model_cfg.Transformer.point_dim,
                              self.box_coder.code_size * self.num_class * self.num_anchors, 4)

        self.up_dimension_geometry = MLP(input_dim=29, hidden_dim=64, output_dim=hidden_dim, num_layers=3)
        self.fuse = MLP(input_dim=hidden_dim * self.num_anchors + (
            (0 if model_cfg.get('USE_POINTNET', False) else 0) + (0 if model_cfg.get('USE_POINTNET', False).get(
                'USE_ABSOLUTE_XYZ', False) else 0)), hidden_dim=model_cfg.Transformer.hidden_dim,
                        output_dim=model_cfg.Transformer.hidden_dim, num_layers=2)
        self.fuse_box = MLP(input_dim=hidden_dim * self.num_anchors, hidden_dim=model_cfg.Transformer.hidden_dim,
                            output_dim=model_cfg.Transformer.hidden_dim, num_layers=2)
        self.up_dimension_back = MLP(input_dim=29, hidden_dim=64, output_dim=hidden_dim, num_layers=3)
        self.up_dimension_motion = MLP(input_dim=30, hidden_dim=64, output_dim=hidden_dim, num_layers=3)
        self.up_dimension_back_motion = MLP(input_dim=30, hidden_dim=64, output_dim=hidden_dim, num_layers=3)
        self.voxel_sampler = build_voxel_sampler(device, return_point_feature=model_cfg.USE_POINTNET)

        self.transformer = build_transformer(model_cfg.Transformer)
        # self.transformer2 = build_transformer(model_cfg.Transformer)
        # self.voxel_sampler = None

        self.class_embed = nn.ModuleList()
        self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim+model_cfg.Transformer.point_dim, self.num_anchors))

        self.bbox_embed = nn.ModuleList()
        for _ in range(self.num_groups):
            self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim+model_cfg.Transformer.point_dim, model_cfg.Transformer.hidden_dim+model_cfg.Transformer.point_dim,
                                       self.box_coder.code_size * self.num_class * self.num_anchors, 4))

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
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

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

    def get_proposal_aware_trajectory_feature(self, src, batch_size, trajectory_rois, num_rois, valid_length):
        proposal_aware_feat_list = []

        for i in range(trajectory_rois.shape[1]):
            corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:, i, :, :].contiguous())

            corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
            corner_points = corner_points.view(batch_size * num_rois, -1)
            trajectory_roi_center = trajectory_rois[:, i, :, :].contiguous().reshape(batch_size * num_rois, -1)[:, :3]
            corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
            proposal_aware_feat = src[:, i * self.num_lidar_points:(i + 1) * self.num_lidar_points, :3].repeat(1, 1,
                                                                                                               9) - \
                                  corner_add_center_points.unsqueeze(1).repeat(1, self.num_lidar_points, 1)

            lwh = trajectory_rois[:, i, :, :].reshape(batch_size * num_rois, -1)[:, 3:6].unsqueeze(1).repeat(1,
                                                                                                             proposal_aware_feat.shape[
                                                                                                                 1], 1)
            diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
            proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))
            proposal_aware_feat_list.append(proposal_aware_feat)

        proposal_aware_feat = torch.cat(proposal_aware_feat_list, dim=1)
        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:, :, 3:]], dim=-1)

        src_gemoetry = self.up_dimension_traj(proposal_aware_feat)

        return src_gemoetry

    def get_proposal_aware_trajectory_motion(self, proxy_point, batch_size, trajectory_rois, num_rois):

        time_stamp = torch.ones([proxy_point.shape[0], proxy_point.shape[1], 1]).cuda()
        padding_zero = torch.zeros([proxy_point.shape[0], proxy_point.shape[1], 2]).cuda()
        point_time_padding = torch.cat([padding_zero, time_stamp], -1)

        num_frames = trajectory_rois.shape[1]

        for i in range(num_frames):
            point_time_padding[:, i * self.num_lidar_points:(i + 1) * self.num_lidar_points, -1] = i * 0.1

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

        proposal_aware_feat = torch.cat([proposal_aware_feat, point_time_padding], -1)
        proxy_point_motion_feat = self.up_dimension_traj_motion(proposal_aware_feat)

        return proxy_point_motion_feat

    def get_proposal_aware_motion_feature(self, proxy_point, batch_size, trajectory_rois, num_rois):

        time_stamp = torch.ones([proxy_point.shape[0], proxy_point.shape[1], 1]).cuda()
        padding_zero = torch.zeros([proxy_point.shape[0], proxy_point.shape[1], 2]).cuda()
        point_time_padding = torch.cat([padding_zero, time_stamp], -1)

        num_frames = trajectory_rois.shape[1]

        for i in range(num_frames):
            point_time_padding[:, i * self.num_lidar_points:(i + 1) * self.num_lidar_points, -1] = i * 0.1

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

        proposal_aware_feat = torch.cat([proposal_aware_feat, point_time_padding], -1)
        proxy_point_motion_feat = self.up_dimension_motion(proposal_aware_feat)

        return proxy_point_motion_feat

    def get_proposal_aware_geometry_feature(self, src, batch_size, trajectory_rois, num_rois):

        proposal_aware_feat_list = []

        for i in range(trajectory_rois.shape[1]):
            corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:, i, :, :].contiguous())

            corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
            corner_points = corner_points.view(batch_size * num_rois, -1)
            trajectory_roi_center = trajectory_rois[:, i, :, :].contiguous().reshape(batch_size * num_rois, -1)[:, :3]
            corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
            proposal_aware_feat = src[:, i * self.num_lidar_points:(i + 1) * self.num_lidar_points, :3].repeat(1, 1,
                                                                                                               9) - \
                                  corner_add_center_points.unsqueeze(1).repeat(1, self.num_lidar_points, 1)

            lwh = trajectory_rois[:, i, :, :].reshape(batch_size * num_rois, -1)[:, 3:6].unsqueeze(1).repeat(1,
                                                                                                             proposal_aware_feat.shape[
                                                                                                                 1], 1)
            diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
            proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))
            proposal_aware_feat_list.append(proposal_aware_feat)
        # time_stamp = [proposal_aware_feat.new_ones(proposal_aware_feat.shape[0],self.num_lidar_points,1)*i*0.1 for i in range(self.num_groups)]
        # time_stamp = torch.concat(time_stamp,1)
        proposal_aware_feat = torch.cat(proposal_aware_feat_list, dim=1)
        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:, :, 3:]], dim=-1)

        src_gemoetry = self.up_dimension_geometry(proposal_aware_feat)

        return src_gemoetry

    def trajectories_auxiliary_branch(self, trajectory_rois):

        time_stamp = torch.ones(
            [trajectory_rois.shape[0], trajectory_rois.shape[1], trajectory_rois.shape[2], 1]).cuda()
        for i in range(time_stamp.shape[1]):
            time_stamp[:, i, :] = i * 0.1

        box_seq = torch.cat([trajectory_rois[:, :, :, :7], time_stamp], -1)

        box_seq[:, :, :, 0:3] = box_seq[:, :, :, 0:3] - box_seq[:, 0:1, :, 0:3]

        roi_ry = box_seq[:, :, :, 6] % (2 * np.pi)
        roi_ry_t0 = roi_ry[:, 0]
        roi_ry_t0 = roi_ry_t0.repeat(1, box_seq.shape[1])

        box_seq = common_utils.rotate_points_along_z(
            points=box_seq.view(-1, 1, box_seq.shape[-1]), angle=-roi_ry_t0.view(-1)
        ).view(box_seq.shape[0], box_seq.shape[1], -1, box_seq.shape[-1])

        box_seq[:, :, :, 6] = 0

        batch_rcnn = box_seq.shape[0] * box_seq.shape[2]

        box_reg, box_feat, _ = self.seqboxembed(
            box_seq.permute(0, 2, 3, 1).contiguous().view(batch_rcnn, box_seq.shape[-1], box_seq.shape[1]))

        return box_reg, box_feat

    def crop_current_frame_points(self, src, batch_size, trajectory_rois, num_rois, batch_dict):

        for bs_idx in range(batch_size):
            cur_batch_boxes = trajectory_rois[bs_idx, 0, :, :7].view(-1, 7)
            cur_radiis = torch.sqrt((cur_batch_boxes[:, 3] / 2) ** 2 + (cur_batch_boxes[:, 4] / 2) ** 2) * 1.1
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]
            dis = torch.norm((cur_points[:, :2].unsqueeze(0) - cur_batch_boxes[:, :2].unsqueeze(1).repeat(1,
                                                                                                          cur_points.shape[
                                                                                                              0], 1)),
                             dim=2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))

            sampled_idx = torch.topk(point_mask.float(), self.num_lidar_points)[1]
            sampled_idx_buffer = sampled_idx[:, 0:1].repeat(1, self.num_lidar_points)
            roi_idx = torch.arange(num_rois)[:, None].repeat(1, self.num_lidar_points)
            sampled_mask = point_mask[roi_idx, sampled_idx]
            sampled_idx_buffer[sampled_mask] = sampled_idx[sampled_mask]

            src[bs_idx] = cur_points[sampled_idx_buffer][:, :, :5]
            empty_flag = sampled_mask.sum(-1) == 0
            src[bs_idx, empty_flag] = 0

        # src = src.repeat([1, 1, trajectory_rois.shape[1], 1])
        src = torch.concat([src, torch.zeros_like(src).repeat(1, 1, trajectory_rois.shape[1] - 1, 1)], dim=-2)
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

    def generate_trajectory_msf(self, cur_batch_boxes, batch_dict):
        num_frames = batch_dict['num_frames']
        trajectory_rois = cur_batch_boxes[:, None, :, :].repeat(1, num_frames, 1, 1)
        trajectory_rois[:, 0, :, :] = cur_batch_boxes
        batch_dict['valid_length'] = torch.ones([batch_dict['batch_size'], num_frames, trajectory_rois.shape[2]])
        # batch_dict['roi_scores'] = batch_dict['roi_scores'][:, :, None].repeat(1, 1, num_frames)

        # simply propagate proposal based on velocity
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = cur_batch_boxes[:, :, 0:2] + i * cur_batch_boxes[:, :, 7:9]
            frame[:, :, 2:] = cur_batch_boxes[:, :, 2:]

            trajectory_rois[:, i, :, :] = frame

        return trajectory_rois

    def generate_trajectory_mppnet(self, cur_batch_boxes, proposals_list, batch_dict):
        num_frames = batch_dict['num_points_all'].shape[1]
        trajectory_rois = cur_batch_boxes[:, None, :, :].repeat(1, num_frames, 1, 1)
        trajectory_rois[:, 0, :, :] = cur_batch_boxes
        valid_length = torch.zeros([batch_dict['batch_size'], num_frames, trajectory_rois.shape[2]])
        valid_length[:, 0] = 1
        num_frames = batch_dict['num_points_all'].shape[1]
        for i in range(1, num_frames):
            trajectory_rois[:, i, :, :2] = trajectory_rois[:, 0, :, :2] + trajectory_rois[:, 0, :, 7:9] * i
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

    def generate_trajectory_denet(self, cur_batch_boxes, proposals_list, batch_dict):
        num_frames = batch_dict['num_frames']
        backward_rois = cur_batch_boxes[:, None, :, :].repeat(1, num_frames, 1, 1)
        backward_rois[:, 0, :, :] = cur_batch_boxes[:, :, :]
        batch_dict['roi_scores'] = batch_dict['roi_scores'][:, :, None].repeat(1, 1, num_frames)
        trajectory_rois = cur_batch_boxes[:, None, :, :].repeat(1, num_frames, 1, 1)
        trajectory_rois[:, 0, :, :] = cur_batch_boxes[:, :, :]
        valid_length = torch.zeros([batch_dict['batch_size'], num_frames, trajectory_rois.shape[2]])
        valid_length[:, 0] = 1

        # simply propagate proposal based on velocity
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes[:, :, :])
            frame[:, :, 0:2] = cur_batch_boxes[:, :, 0:2] + i * cur_batch_boxes[:, :, 7:9]
            frame[:, :, 2:] = cur_batch_boxes[:, :, 2:]

            backward_rois[:, i, :, :] = frame

            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = trajectory_rois[:, i - 1, :, 0:2] + trajectory_rois[:, i - 1, :, 7:9]
            frame[:, :, 2:] = trajectory_rois[:, i - 1, :, 2:]

            for bs_idx in range(batch_dict['batch_size']):
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame.cuda()[bs_idx, :, :7],
                                                        proposals_list.cuda()[bs_idx, i, :, :7]).to(device)

                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)

                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                bg_inds = ((max_overlaps < 0.5).nonzero().view(-1))
                valid_length[bs_idx, i, fg_inds] = 1
                trajectory_rois[bs_idx, i, fg_inds, :] = proposals_list[bs_idx, i, traj_assignment[fg_inds]]
                trajectory_rois[bs_idx, i, bg_inds, :] = frame[bs_idx, bg_inds, :]

            batch_dict['valid_length'] = valid_length

        return backward_rois, trajectory_rois


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        num_frames = batch_dict['num_points_all'].shape[-1]
        roi_scores = batch_dict['roi_scores'][:, 0, :]
        batch_dict['roi_scores'] = roi_scores[:, roi_scores.sum(0) > 0]
        # batch_dict['proposals_list'] = batch_dict['roi_boxes']
        batch_dict['roi_boxes'] = batch_dict['roi_boxes'][:, 0, :][:, roi_scores.sum(0) > 0]
        batch_dict['roi_labels'] = batch_dict['roi_labels'][:, 0, :][:, roi_scores.sum(0) > 0].long()
        batch_dict['num_frames'] = batch_dict['num_points_all'].shape[-1]
        batch_dict['num_anchors'] = self.model_cfg.NUM_ANCHORS
        num_rois = batch_dict['roi_boxes'].shape[1]
        batch_size = batch_dict['batch_size']
        cur_batch_boxes = copy.deepcopy(batch_dict['roi_boxes'].detach())
        # cur_batch_boxes = torch.concat([cur_batch_boxes,batch_dict['roi_labels'][:,:,None].float()],dim=-1)
        batch_dict['cur_frame_idx'] = 0
        # proposals_list = batch_dict['proposals_list']
        self.anchor_sampler = build_voxel_sampler_anchor(roi_scores.device)
        num_sample = self.num_lidar_points
        if not self.model_cfg.get('PRE_AUG', False):
            # anchors_rois = self.anchor_sampler(batch_size,
            #                                    torch.concat([cur_batch_boxes, batch_dict['roi_labels'][:, :, None]],
            #                                                 dim=-1), num_sample, batch_dict['roi_scores'], batch_dict,
            #                                    num_anchors=batch_dict['num_anchors'], return_boxes=True)
            # anchors_rois = anchors_rois.transpose(1,2)
            anchors_rois = batch_dict['anchors']
            trajectory_rois = self.generate_trajectory_msf(anchors_rois.reshape(batch_size, -1, anchors_rois.shape[-1]),
                                                           batch_dict)
            # trajectory_rois = self.generate_trajectory_msf(cur_batch_boxes, batch_dict)
            # batch_dict['backward_rois'] = backward_rois
            batch_dict['trajectory_rois'] = trajectory_rois
            batch_dict['traj_memory'] = trajectory_rois
            # batch_dict['roi_labels'] = backward_rois[:,0,:,-1].long()
            # batch_dict['roi_boxes'] = batch_dict['backward_rois'][:, 0]

        # trajectory_rois,valid = self.generate_trajectory_mppnet(cur_batch_boxes,proposals_list, batch_dict)
        # batch_dict['traj_memory'] = trajectory_rois
        batch_dict['has_class_labels'] = True
        # batch_dict['trajectory_rois'] = trajectory_rois

        # if self.voxel_sampler is None:
        #     self.voxel_sampler = build_voxel_sampler(device)
        #     self.voxel_sampler_traj = build_voxel_sampler_traj(device)
        # src1 = self.voxel_sampler(batch_size, backward_rois, num_sample, batch_dict)
        if not self.training:
            # mask = (src1[0,:,:128,0]!=0).sum(-1)>0
            batch_dict['pred_anchors'] = anchors_rois
            # batch_dict['batch_cls_preds'] = batch_dict['roi_scores'][0,:,0][mask][None,:,None]
            # batch_dict['cls_preds_normalized'] = True
            # batch_dict['roi_labels'] = batch_dict['roi_labels'][0][mask][None,:]
            return batch_dict
        if self.training:
            if not self.model_cfg.get('PRE_AUG', False):
                targets_dict = self.assign_targets(batch_dict)
            else:
                targets_dict = batch_dict['targets_dict']
            batch_dict['roi_boxes'] = targets_dict['rois']
            # batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # batch_dict['roi_scores'] = targets_dict['roi_scores']
            targets_dict['trajectory_rois'][:, batch_dict['cur_frame_idx'], :, :] = batch_dict['roi_boxes']
            # trajectory_rois = targets_dict['trajectory_rois']
            trajectory_rois = targets_dict['trajectory_rois']
            empty_mask = batch_dict['roi_boxes'][:,
                         torch.arange(batch_dict['roi_boxes'].shape[1] // self.num_anchors) * self.num_anchors, :6].sum(
                -1) == 0
            valid_length = targets_dict['valid_length']
        else:
            empty_mask = batch_dict['roi_boxes'][:, :, :6].sum(-1) == 0
            batch_dict['valid_traj_mask'] = ~empty_mask
            batch_dict['roi_boxes'] = backward_rois[:, 0]
            # valid_length = batch_dict['valid_length'].to(device)
        rois = batch_dict['roi_boxes']
        num_rois = batch_dict['roi_boxes'].shape[1]
        num_anchors = batch_dict['num_anchors']

        trajectory_rois = trajectory_rois.reshape(batch_size, num_frames, -1, batch_dict['num_anchors'],
                                                  trajectory_rois.shape[-1])
        signal = True
        if signal:
            src1, src1_features,query_points_features = self.voxel_sampler(batch_size, torch.mean(trajectory_rois, dim=-2), num_sample,
                                                     batch_dict)
            src1,src_idx1 =src1[...,:-1], src1[...,-1].long()
            batch_dict['src_idx1'] = src_idx1.view(batch_size*num_rois//num_anchors,num_frames,-1).permute(2,1,0).reshape(-1,batch_size*num_rois//num_anchors*num_frames)
            batch_dict['query_points_features1'] = query_points_features
            src1 = src1[:, :, None, :, :].repeat(1, 1, batch_dict['num_anchors'], 1, 1).view(batch_size * num_rois, -1,
                                                                                             src1.shape[-1])
            # src2 = src2.view(batch_size * num_rois,-1,src2.shape[-1])
            # xyz1 = src1[:, :, :3]
            trajectory_rois = trajectory_rois.view(batch_size, num_frames, -1, trajectory_rois.shape[-1])
            # xyz2 = src2[:, :, :3]
            src_trajectory_feature = self.get_proposal_aware_geometry_feature(src1, batch_size, trajectory_rois,
                                                                              num_rois)
            # src_trajectory_feature = self.get_proposal_aware_geometry_feature(src2, batch_size, trajectory_rois, num_rois)

            src_motion_feature = self.get_proposal_aware_motion_feature(src1, batch_size, trajectory_rois, num_rois)
            # src_motion_feature2 = self.get_proposal_aware_motion_feature(src2,batch_size,trajectory_rois,num_rois)

            src1 = src_trajectory_feature + src_motion_feature
            # src2 = src_trajectory_feature+src_motion_feature2
            src1 = src1.reshape(-1, batch_dict['num_anchors'], src1.shape[-2], src1.shape[-1])
            src1_features = src1_features.view(batch_size * num_rois // num_anchors, -1, src1_features.shape[-1])
            # if self.model_cfg.get('USE_POINTNET', False):
            #     src1 = self.fuse(torch.concat(
            #         [src1.transpose(1, 2).reshape(-1, src1.shape[-2], self.num_anchors * self.hidden_dim),
            #          src1_features], dim=-1))
            # else:
            #     src1 = self.fuse(src1.transpose(1, 2).reshape(-1, src1.shape[-2], self.num_anchors * self.hidden_dim))
                # src1 = src1.reshape(-1,src1.shape[-2],self.num_anchors*self.hidden_dim)
            src1 = self.fuse(src1.transpose(1, 2).reshape(-1, src1.shape[-2], self.num_anchors * self.hidden_dim))

            # num_rois_all = src1.shape[0]
            # src = src_geometry_feature + src_motion_feature
            # src = self.conv(torch.concat([src_trajectory_feature,src_backward_feature],dim=-1).permute(0,2,1)).permute(0,2,1)
            box_reg, feat_box = self.trajectories_auxiliary_branch(trajectory_rois)
            feat_box = self.fuse_box(feat_box.reshape(-1, self.num_anchors * feat_box.shape[-1]))
            if self.model_cfg.get('USE_TRAJ_EMPTY_MASK', None):
                src1[empty_mask.view(-1)] = 0
                # src2[empty_mask.view(-1)] = 0

            hs1, tokens1 = self.transformer(src1,src1_features,batch_dict,pos=None)
            # hs2,tokens2 = self.transformer(src2,pos=None)
            # hs2,tokens2 = self.transformer2(src2,pos=None)
            # hs = hs[:,:num_rois_all]
        else:

            src = self.voxel_sampler(batch_size, trajectory_rois, num_sample, batch_dict)

            src = src.view(batch_size * num_rois, -1, src.shape[-1])

            src_geometry_feature = self.get_proposal_aware_geometry_feature(src, batch_size, trajectory_rois, num_rois)

            src_motion_feature = self.get_proposal_aware_motion_feature(src, batch_size, trajectory_rois, num_rois)

            src = src_geometry_feature + src_motion_feature

            box_reg, feat_box = self.trajectories_auxiliary_branch(trajectory_rois)

            if self.model_cfg.get('USE_TRAJ_EMPTY_MASK', None):
                src[empty_mask.view(-1)] = 0

            hs1, tokens1 = self.transformer(src, pos=None)

        point_cls_list = []
        point_reg_list = []

        for i in range(self.num_enc_layer):
            point_cls_list.append(self.class_embed[0](tokens1[i][0]))

        for i in range(hs1.shape[0]):
            for j in range(self.num_enc_layer):
                point_reg_list.append(self.bbox_embed[i](tokens1[j][i]))

        point_cls = torch.cat(point_cls_list, 0)

        point_reg = torch.cat(point_reg_list, 0)
        # hs2 = hs2.permute(1,0,2).reshape(hs2.shape[1],-1)
        hs1 = hs1.permute(1, 0, 2).reshape(hs1.shape[1], -1)

        joint_reg = self.jointembed(torch.cat([hs1, feat_box], -1))

        rcnn_cls = point_cls
        rcnn_reg = joint_reg

        if not self.training:
            rcnn_cls = rcnn_cls[-rcnn_cls.shape[0] // self.num_enc_layer:]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['roi_boxes'], cls_preds=rcnn_cls,
                box_preds=rcnn_reg
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
            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_box_cls_layer_loss1(self, forward_ret_dict):
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

    def get_box_reg_layer_loss1(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)

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
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )
            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                            loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if self.model_cfg.USE_AUX_LOSS:
                point_reg = forward_ret_dict['point_reg']

                groups = point_reg.shape[0] // reg_targets.shape[0]
                if groups != 1:
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                            point_reg[i * slice:(i + 1) * slice].view(slice, -1).unsqueeze(dim=0),
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
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
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
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )
            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] * \
                            loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if self.model_cfg.USE_AUX_LOSS:
                point_reg = forward_ret_dict['point_reg']
                point_reg = point_reg.view(-1, 7)
                groups = point_reg.shape[0] // reg_targets.shape[0]
                if groups != 1:
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                            point_reg[i * slice:(i + 1) * slice].view(slice, -1).unsqueeze(dim=0),
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
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
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
            # rcnn_cls_flat = rcnn_cls
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
