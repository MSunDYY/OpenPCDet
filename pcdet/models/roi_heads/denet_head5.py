import os.path
from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.functional as F
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.msf_utils import TransformerEncoderLayer,_get_activation_fn
from ..model_utils.denet5_utils import build_transformer,unflatten,SpatialMixerBlockCompress,CrossMixerBlock,SpatialMixerBlock,SpatialDropBlock
from ..model_utils.denet5_utils import build_voxel_sampler,PointNet,MLP
from .msf_head import ProposalTargetLayerMPPNet
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_modules import PointnetSAModuleMSG
from pcdet import device
from pathlib import Path
import os
import time
class ProposalTargetLayerMPPNet1(ProposalTargetLayer):
    def __init__(self, roi_sampler_cfg):
        super().__init__(roi_sampler_cfg = roi_sampler_cfg)

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
        batch_backward_rois,batch_trajectory_rois,batch_valid_length = self.sample_rois_for_mppnet(batch_dict=batch_dict)

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
                        'gt_iou_of_rois': batch_roi_ious,'roi_scores': batch_roi_scores,
                        'roi_labels': batch_roi_labels,'reg_valid_mask': reg_valid_mask, 
                        'rcnn_cls_labels': batch_cls_labels,'trajectory_rois':batch_trajectory_rois,
                        'backward_rois':batch_backward_rois,
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
        rois = batch_dict['backward_rois'][:, cur_frame_idx, :, :]
        roi_scores = batch_dict['roi_scores'][:, :, cur_frame_idx]
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, gt_boxes.shape[-1])
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        backward_rois = batch_dict['backward_rois']
        trajectory_rois = batch_dict['trajectory_rois']
        batch_trajectory_rois = rois.new_zeros(batch_size, trajectory_rois.shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE,
                                                trajectory_rois.shape[-1])
        batch_backward_rois = rois.new_zeros(batch_size, backward_rois.shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE,
                                               backward_rois.shape[-1])

        valid_length = batch_dict['valid_length']
        batch_valid_length = rois.new_zeros(
            (batch_size, batch_dict['backward_rois'].shape[1], self.roi_sampler_cfg.ROI_PER_IMAGE))

        for index in range(batch_size):

            cur_backward_rois = backward_rois[index]
            cur_trajectory_rois = trajectory_rois[index]
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = rois[index], gt_boxes[index], roi_labels[index], \
            roi_scores[index]

            if 'valid_length' in batch_dict.keys():
                cur_valid_length = valid_length[index].to(device)

            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1

            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )

            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
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

            if self.roi_sampler_cfg.USE_TRAJ_AUG.ENABLED:
                batch_backward_rois_list = []
                batch_trajectory_rois_list = []
                for idx in range(0, batch_dict['num_frames']):
                    if idx == cur_frame_idx:
                        batch_backward_rois_list.append(
                            cur_backward_rois[cur_frame_idx:cur_frame_idx + 1, sampled_inds])
                        batch_trajectory_rois_list.append(
                            cur_trajectory_rois[cur_frame_idx:cur_frame_idx+1,sampled_inds]
                        )
                        continue
                    fg_backs, _ = self.aug_roi_by_noise_torch(cur_backward_rois[idx, fg_inds],
                                                              cur_backward_rois[idx, fg_inds][:, :8],
                                                              max_overlaps[fg_inds], \
                                                              aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES,
                                                              pos_thresh=self.roi_sampler_cfg.USE_TRAJ_AUG.THRESHOD)
                    bg_backs = cur_backward_rois[idx, bg_inds]
                    fg_trajs,_ = self.aug_roi_by_noise_torch(cur_trajectory_rois[idx,fg_inds],
                                                             cur_trajectory_rois[idx,fg_inds][:,:8],
                                                             max_overlaps[fg_inds],
                                                             aug_times=self.roi_sampler_cfg.ROI_FG_AUG_TIMES,
                                                             pos_thresh=self.roi_sampler_cfg.USE_TRAJ_AUG.THRESHOD
                                                             )
                    bg_trajs = cur_trajectory_rois[idx,bg_inds]

                    batch_backward_rois_list.append(torch.cat([fg_backs, bg_backs], 0)[None, :, :])
                    batch_trajectory_rois_list.append(torch.cat([fg_trajs,bg_trajs],0)[None,:,:])
                batch_backward_rois[index] = torch.cat(batch_backward_rois_list, 0)
                batch_trajectory_rois[index] = torch.cat(batch_trajectory_rois_list,0)
            else:
                batch_backward_rois[index] = cur_backward_rois[:, sampled_inds]
                batch_trajectory_rois[index] = cur_trajectory_rois[:,sampled_inds]
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels, batch_backward_rois,batch_trajectory_rois, batch_valid_length


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
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(aug_box3d[:,:7], gt_box3d[:,:7])
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def random_aug_box3d(self,box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """

        if self.roi_sampler_cfg.REG_AUG_METHOD == 'single':
            pos_shift = (torch.rand(3, device=box3d.device) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]
            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot, box3d[7:]], dim=0)
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


class MSFEncoderLayer(nn.Module):
    count = 0

    def __init__(self,  d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_points=None, num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count

        self.num_point = num_points
        self.num_groups = num_groups
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.cross_conv_1 = nn.Linear(d_model * 2, d_model)
        self.cross_norm_1 = nn.LayerNorm(d_model)
        self.cross_conv_2 = nn.Linear(d_model * 2, d_model)
        self.cross_norm_2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(
            d_model,
        )

    def forward(self,src,pos = None):

        src_intra_group_fusion,weight = self.mlp_mixer_3d(src[1:])
        src = torch.cat([src[:1], src_intra_group_fusion], 0)

        token = src[:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion

        src_summary = self.self_attn(token, key, value=src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = torch.cat([token, src[1:]], 0)
        # if self.config.get('SHRINK_POINTS',False):
        #     sampled_inds =torch.topk(weight.sum(1),k=weight.shape[1]//2)[1]

            # src = torch.cat([token,torch.gather(src[1:],0,sampled_inds.transpose(0,1)[:,:,None].repeat(1,1,src.shape[-1]))],dim=0)

        src_all_groups = src[1:].view((src.shape[0] - 1) * self.num_groups, -1, src.shape[-1])
        src_groups_list = src_all_groups.chunk(self.num_groups, 0)
        # src_groups_list = [src_all_groups[torch.arange(src_all_groups.shape[0]//self.num_groups) * self.num_groups+i] for i in range(self.num_groups)]
        src_all_groups = torch.stack(src_groups_list, 0)

        src_max_groups = torch.max(src_all_groups, 1, keepdim=True).values
        src_past_groups = torch.cat([src_all_groups[1:], \
                                     src_max_groups[:-1].repeat(1, (src.shape[0] - 1), 1, 1)], -1)
        src_all_groups[1:] = self.cross_norm_1(self.cross_conv_1(src_past_groups) + src_all_groups[1:])

        src_max_groups = torch.max(src_all_groups, 1, keepdim=True).values
        src_past_groups = torch.cat([src_all_groups[:-1], \
                                     src_max_groups[1:].repeat(1, (src.shape[0] - 1), 1, 1)], -1)
        src_all_groups[:-1] = self.cross_norm_2(self.cross_conv_2(src_past_groups) + src_all_groups[:-1])

        src_inter_group_fusion = src_all_groups.permute(1, 0, 2, 3).contiguous().flatten(1, 2)

        src = torch.cat([src[:1], src_inter_group_fusion], 0)

        return src, src[0].unflatten(0,(-1,4)).transpose(0,1)

    def forward_pre(self, src,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class KPTransformer(nn.Module):
    def __init__(self,model_cfg=None):
        super(KPTransformer, self).__init__()
        self.spatial_temporal_type = model_cfg.SPATIAL_TEMPORAL_TYPE
        self.channels = model_cfg.hidden_dim
        self.num_frames = model_cfg.num_frames
        self.num_groups = model_cfg.num_groups
        self.drop_rate = model_cfg.drop_rate
        self.Attention = SpatialDropBlock(self.channels,dropout=0.1,batch_first=True)
        self.Attention2 = SpatialDropBlock(self.channels,dropout=0.1,batch_first=True)
        self.Attention3 = SpatialMixerBlock(self.channels,dropout=0.1,batch_first=True)
        # self.Crossatten1= CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)
        # self.Crossatten2 = CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)

        # self.decoder_layer1 = CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)
        self.decoder_layer2 = CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)
        self.decoder_layer3 = CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)


        self.conv1 = nn.Sequential(
            nn.Conv1d(self.channels*self.num_frames//self.num_groups,self.channels,1,1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Conv1d(self.channels*2,self.channels,1)
        )
        self.linear1 = nn.ModuleList([nn.Linear(self.channels*2,self.channels) for _ in range(self.num_frames//self.num_groups)])
        self.dropout1 = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.num_frames//self.num_groups)])
        self.dropout2 = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.num_groups)])

        self.norm1 = nn.LayerNorm(self.channels)

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.channels *self.num_groups, self.channels , 1, 1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Conv1d(self.channels*2,self.channels,1)
        )
        self.linear2 = nn.ModuleList([nn.Linear(self.channels*2 , self.channels) for _ in range(self.num_groups)])
        self.norm2 = nn.LayerNorm(self.channels)
        if self.spatial_temporal_type=='BiFA':
            self.Attention2 = MSFEncoderLayer(d_model=self.channels,dim_feedforward=512,nhead=8,num_groups=4)
            self.Attention3 = MSFEncoderLayer(d_model=self.channels,dim_feedforward=512,nhead=8,num_groups=4)
            # self.decoder_layer2 = SpatialMixerBlock(channels=self.channels,num_heads=8,batch_first=True)




    def forward(self, src,token,src_cur):
        # src = src.permute(2,1,0,3).flatten(1,2)
        B = src_cur.shape[0]
        token_list = list()
        src = src.reshape(src.shape[0]*self.num_frames,-1,src.shape[-1])
        num_frames_single_group = self.num_frames // self.num_groups
        src,weight,sampled_inds = self.Attention(src,return_weight=True,drop=self.drop_rate[0])
        src = src.reshape(B,self.num_frames,-1,self.channels)

        signal = True
        if signal:
            src=src.unflatten(1,(-1,self.num_groups)).transpose(1,2).flatten(0,1)
            src_max = src.max(2).values
            src_max = src_max.flatten(1,2)
            src_max = self.conv1(src_max.unsqueeze(-1)).squeeze()
            src_new = [self.dropout1[i](self.linear1[i](torch.concat([src[:,i],src_max[:,None,:].repeat(1,src.shape[2],1)],dim=-1))) for i in range(self.num_frames//self.num_groups)]

            src = self.norm1(src + torch.stack(src_new,1)).flatten(1,2)
        else:
            src = src.unflatten(1,(-1,self.num_groups)).transpose(1,2).flatten(0,1).flatten(1,2)
        # token1 = self.decoder_layer2(token,src.unflatten(0,(B,-1))[:,0])
        # token_list.append(token1)

        if self.spatial_temporal_type=='BiFA':

            src = torch.concat([token.repeat(1,4,1).reshape(B*4,1,-1),src],dim=1).transpose(0,1)

            src,token2 = self.Attention2(src)
            src,token3 = self.Attention3(src)
            token_list.append(token2.transpose(0,1))
            token_list.append(token3.transpose(0,1))
            return token_list

        src,weight,sampled_inds = self.Attention2(src,return_weight=True,drop=self.drop_rate[1])

        if signal:
            src = src.unflatten(0,(-1,self.num_groups))
            src_max = src.max(2).values
            src_max = src_max.flatten(1,2)
            src_max = self.conv2(src_max.unsqueeze(-1)).squeeze(-1)
            # src = src.flatten(1,2)

            src_new = [self.dropout2[i](self.linear2[i](torch.concat([src[:,i],src_max[:,None,:].repeat(1,src.shape[2],1)],dim=-1))) for i in range(self.num_groups)]

            src = self.norm2(src + torch.stack(src_new,1)).flatten(1,2)
        else:
            src = src.reshape(-1,self.num_groups*src.shape[1],src.shape[-1])
        src = self.Attention3(src,return_weight = False)

        # src_cur = self.Crossatten2(src_cur,src_new)
        token = self.decoder_layer3(token,src)
        token_list.append(token)
        # src = self.pointnet(src.permute(0,2,1))
        # # src = src.permute(0,2,1)
        # x = torch.max(src,dim=-1).values
        #
        # x = F.relu(self.x_bn1(self.fc1(x)))
        # feat = F.relu(self.x_bn2(self.fc2(x)))
        #
        # centers = self.fc_ce2(F.relu(self.fc_ce1(feat)))
        # sizes = self.fc_s2(F.relu(self.fc_s1(feat)))
        # headings = self.fc_hr2(F.relu(self.fc_hr1(feat)))

        return token_list

class Pointnet(nn.Module):
    def __init__(self,channels):
        super(Pointnet, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x,dim=-1).values
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PointNetLK(nn.Module):
    def __init__(self,channels):
        super(PointNetLK, self).__init__()
        self.pointnet = Pointnet(channels)

    def forward(self, source, target):
        source_feature = self.pointnet(source)
        target_feature = self.pointnet(target)
        transform = self.estimate_transform(source_feature, target_feature)
        return transform

    def estimate_transform(self, source_feature, target_feature):
        # 简单的线性变换估计
        transform = target_feature - source_feature
        return transform

class DENet5Head(RoIHeadTemplate):
    def __init__(self,model_cfg, num_class=1,**kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.proposal_target_layer = ProposalTargetLayerMPPNet(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.use_time_stamp = self.model_cfg.get('USE_TIMESTAMP',None)
        self.num_lidar_points = self.model_cfg.Transformer.num_lidar_points
        self.num_lidar_points2 = self.model_cfg.Transformer2st.num_lidar_points
        self.avg_stage1_score = self.model_cfg.get('AVG_STAGE1_SCORE', None)
        self.nhead = model_cfg.Transformer.nheads
        self.num_enc_layer = model_cfg.Transformer.enc_layers
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.num_groups = model_cfg.Transformer.num_groups
        self.voxel_sampler_cur = build_voxel_sampler(device, return_point_feature=model_cfg.USE_POINTNET)
        self.voxel_sampler = build_voxel_sampler(device)
        self.grid_size = model_cfg.ROI_GRID_POOL.GRID_SIZE
        # self.pos_embding = nn.Linear(4,128)
        # self.cross = nn.ModuleList([CrossAttention(3,4,256,None) for i in range(4)])
        self.seqboxembed = PointNet(10,model_cfg=self.model_cfg)
        self.memory_num = list()
        self.delay = list()
        self.jointembed = MLP(self.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4)

        self.up_dimension_geometry = MLP(input_dim = 29, hidden_dim = 64, output_dim =hidden_dim, num_layers = 3)
        self.up_dimension_motion = MLP(input_dim = 30, hidden_dim = 64, output_dim =hidden_dim, num_layers = 3)

        self.transformer = build_transformer(model_cfg.Transformer)
        self.transformer2st = KPTransformer(model_cfg.Transformer2st)

        self.class_embed = nn.ModuleList()
        self.class_embed_final = nn.Linear(model_cfg.Transformer2st.hidden_dim,1)
        self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim, 1))
        # self.points_feature_cls = nn.Sequential(
        #     nn.Linear(131,256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256,3)
        # )

        if model_cfg.Transformer2st.SPATIAL_TEMPORAL_TYPE=='BiFA':
            self.num_groups=4
            self.jointembed = MLP(self.hidden_dim*4,model_cfg.Transformer.hidden_dim,self.box_coder.code_size*self.num_class,4)
        self.bbox_embed = nn.ModuleList()
        # self.bbox_embed_final = MLP(model_cfg.Transformer.hidden_dim,model_cfg.Transformer.hidden_dim,self.box_coder.code_size * self.num_class,4)
        for _ in range(self.num_groups):
            self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4))

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

    def get_corner_points_of_roi(self, rois,with_center=False):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)
        local_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()

        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        if with_center:
            global_roi_grid_points = torch.concat([global_roi_grid_points,global_center[:,None,:]],dim=1)
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
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim = -1)
        return src
    def points_features_pool(self, roi_boxes, query_points,query_boxes_idx, num_sample,):

        sampled_points_list = []
        sampled_points_features_list = []
        for idx in range(query_boxes_idx.shape[0]-1):
            cur_points = query_points[query_boxes_idx[idx]:query_boxes_idx[idx+1]]

            cur_boxes = roi_boxes[idx]
            if len(cur_points) < num_sample:
                cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])
            cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1)
            dis = torch.norm(
                (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))
            # valid_points_mask = (point_mask.sum(0))!=0
            # cur_points,point_mask = cur_points[valid_points_mask],point_mask[:,valid_points_mask]

            sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample, 1)



            sampled_mask = sampled_mask.bool()
            sampled_idx_ = (sampled_idx*sampled_mask).view(-1, 1).repeat(1, cur_points.shape[-1])
            sampled_points = torch.gather(cur_points, 0, sampled_idx_).view(len(sampled_mask), num_sample, -1)
            # idx = idx * sampled_mask + idx_checkpoint
            sampled_points_features_list.append(sampled_points[:,:,3:])
            sampled_points = (sampled_points[:,:,:3]-cur_boxes[:,None,:3])/cur_radiis[:,None,None]
            sampled_points_list.append(sampled_points)

        sampled_points = torch.concat(sampled_points_list, 0)
        sampled_points_features = torch.concat(sampled_points_features_list,0)
        return sampled_points.transpose(0,1),sampled_points_features.transpose(0,1)


    def get_proposal_aware_motion_feature(self, proxy_point,  trajectory_rois,valid):
        num_rois = proxy_point.shape[0]
        padding_mask = proxy_point[:,:,0:1]!=0
        time_stamp = torch.ones([proxy_point.shape[0], proxy_point.shape[1], 1],device = device)
        padding_zero = torch.zeros([proxy_point.shape[0], proxy_point.shape[1], 2],device = device)
        point_time_padding = torch.cat([padding_zero, time_stamp], -1)

        num_frames = trajectory_rois.shape[1]
        num_points_single_frame = proxy_point.shape[1]//num_frames
        for i in range(num_frames):
            point_time_padding[:, i * num_points_single_frame : (i+1) * num_points_single_frame, -1] = i * 0.1
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois.contiguous())

        corner_points = corner_points.flatten(-2,-1)
        trajectory_roi_center = trajectory_rois.flatten(0,1)[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)

        lwh = trajectory_rois.flatten(0,1)[:, 3:6].unsqueeze(1)
        diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5

        if True:
            motion_aware_feat = proxy_point[:,:,:3].repeat(1,1,9)-corner_add_center_points.unflatten(0,(num_rois,-1))[:,0:1]
        else:
            motion_aware_feat = corner_add_center_points.unflatten(0,(num_rois,-1))-corner_add_center_points.unflatten(0,(num_rois,-1))[:,0:1]
            motion_aware_feat = motion_aware_feat.unsqueeze(2).repeat(1,1,num_points_single_frame,1).flatten(1,2)
        geometry_aware_feat = proxy_point.reshape(num_rois*num_frames,num_points_single_frame,-1)[:, :, :3].repeat(1, 1, 9) - corner_add_center_points.unsqueeze(1)
        motion_aware_feat = self.spherical_coordinate(motion_aware_feat, diag_dist=diag_dist.unflatten(0,(num_rois,-1))[:,:1,:])
        geometry_aware_feat = self.spherical_coordinate(geometry_aware_feat,diag_dist=diag_dist[:,:,None])

        # motion_aware_feat = self.up_dimension_motion(torch.cat([motion_aware_feat, point_time_padding,valid.transpose(0,1)[:,:,None].repeat(1,1,num_points_single_frame).reshape(num_rois,-1,1)], -1))
        motion_aware_feat = self.up_dimension_motion(torch.cat([motion_aware_feat, point_time_padding], -1))

        geometry_aware_feat = self.up_dimension_geometry(torch.concat([geometry_aware_feat.reshape(num_rois,num_frames*num_points_single_frame,-1),proxy_point[:,:,3:]],-1))

        return motion_aware_feat + geometry_aware_feat

    def get_proposal_aware_geometry_feature(self, src, trajectory_rois):
        padding_mask = src[:,:,0:1]!=0

        num_rois = trajectory_rois.shape[0]

        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois.contiguous())

        # corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
        corner_points = corner_points.view(num_rois, -1)
        trajectory_roi_center = trajectory_rois[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
        proposal_aware_feat = src[:,:,:3].repeat(1, 1,9) - corner_add_center_points.unsqueeze(1).repeat(1, src.shape[1], 1)

        lwh = trajectory_rois[:, 3:6].unsqueeze(1).repeat(1,proposal_aware_feat.shape[1], 1)
        diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
        proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))
            # proposal_aware_feat_list.append(proposal_aware_feat)
        # time_stamp = [proposal_aware_feat.new_ones(proposal_aware_feat.shape[0],self.num_lidar_points,1)*i*0.1 for i in range(self.num_groups)]
        # time_stamp = torch.concat(time_stamp,1)
        # proposal_aware_feat = torch.cat(proposal_aware_feat_list, dim=1)
        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:, :, 3:]], dim=-1)
        # proposal_aware_feat = proposal_aware_feat*padding_mask
        src_gemoetry = self.up_dimension_geometry(proposal_aware_feat)

        return src_gemoetry

    def trajectories_auxiliary_branch(self,trajectory_rois,valid_length):

        # time_stamp = torch.ones([trajectory_rois.shape[0],trajectory_rois.shape[1],1]).cuda()
        # for i in range(time_stamp.shape[1]):
        #     time_stamp[:,i,:] = i*0.1
        batch_rcnn = trajectory_rois.shape[0]
        time_stamp = torch.arange(trajectory_rois.shape[1],device=device)[None,:,None].repeat(batch_rcnn,1,1)

        box_seq = torch.cat([trajectory_rois[:,:,:7],time_stamp,(valid_length>0).unsqueeze(-1).float(),(valid_length==0).unsqueeze(-1).float()],-1)

        box_seq[:, :, 0:3] = box_seq[:, :, 0:3] - box_seq[:, 0:1,  0:3]

        roi_ry = box_seq[:,:,6] % (2 * np.pi)
        roi_ry_t0 = roi_ry[:,0] 
        roi_ry_t0 = roi_ry_t0


        box_seq = common_utils.rotate_points_along_z(
            points=box_seq, angle=-roi_ry_t0.view(-1)
        ).view(box_seq.shape[0],box_seq.shape[1],  box_seq.shape[-1])

        box_seq[:, :, 6] = (box_seq[:,:,6 ] -box_seq[:,0:1,6])%(2*np.pi)


        box_reg, box_feat, _ = self.seqboxembed(box_seq.permute(0,2,1).contiguous())
        
        return box_reg, box_feat

    
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
    
    def encode_torch(self,gt_boxes,points):
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)
        from pcdet.utils.common_utils import rotate_points_along_z
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        
        points = rotate_points_along_z((points-gt_boxes[:,:3])[:,None,:],angle=-1 * gt_boxes[:,6]).squeeze()
        xa, ya, za = torch.split(points, 1, dim=-1)
        xt = xa/dxg
        yt = ya/dyg
        zt = za/dzg
        

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)
    def assign_stack_targets(self, points, gt_boxes,points_bs_idx, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        from pcdet.utils import box_utils
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
        extend_gt_boxes = box_utils.enlarge_box3d(gt_boxes.view(-1,gt_boxes.shape[-1]),extra_width=[0.2,0.2,0.2]).unflatten(0,(gt_boxes.shape[0],-1))
        points_bs_idx = F.pad(torch.cumsum(points_bs_idx,0),(1,0),value=0)
        # assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        # assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
        #     'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        # bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 5)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        
        for k in range(batch_size):
           
            points_single = points[points_bs_idx[k]:points_bs_idx[k+1]]
            point_cls_labels_single = point_cls_labels.new_zeros(points_bs_idx[k+1]-points_bs_idx[k])
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = gt_box_of_fg_points[:, -1].long()
            point_cls_labels[points_bs_idx[k]:points_bs_idx[k+1]] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros(points_bs_idx[k+1]-points_bs_idx[k], 5)
                fg_point_box_labels = self.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :7], points=points_single[fg_flag],
                    # gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[points_bs_idx[k]:points_bs_idx[k+1]] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def generate_trajectory_mppnet(self, cur_batch_boxes, proposals_list, batch_dict):
        num_frames = batch_dict['num_points_all'].shape[1]
        trajectory_rois = cur_batch_boxes[:, None, :, :].repeat(1, num_frames, 1, 1)
        trajectory_rois[:, 0, :, :] = cur_batch_boxes
        valid_length = torch.zeros([batch_dict['batch_size'], num_frames, trajectory_rois.shape[2]])
        valid_length[:, 0] = 1
        num_frames = batch_dict['num_points_all'].shape[1]
        for i in range(1, num_frames):
            trajectory_rois[:,i,:,:2]=trajectory_rois[:,0,:,:2]+trajectory_rois[:,0,:,7:9]*i
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
        # trajectory_rois[:,:-1,:,7:9] = trajectory_rois[:,1:,:,:2]-trajectory_rois[:,:-1,:,7:9]
        batch_dict['valid_length'] = valid_length

        return trajectory_rois, valid_length

    @staticmethod
    def get_corner_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx / (grid_size-1) * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

       
    def pos_offset_encoding(self,src,boxes):
        radiis = torch.norm(boxes[:,3:5]/2,dim=-1)
        return (src-boxes[None,:,:3])/radiis[None,:,None].repeat(1,1,3)

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        st = time.time()
        num_frames = self.model_cfg.Transformer2st.num_frames
        roi_scores = batch_dict['roi_scores'][:, 0, :]

        batch_dict['roi_scores'] = roi_scores[:,roi_scores.sum(0)>=0]
        batch_dict['proposals_list'] = batch_dict['roi_boxes']
        batch_dict['roi_boxes'] = batch_dict['roi_boxes'][:,0,:][:,roi_scores.sum(0)>=0]
        batch_dict['roi_labels'] = batch_dict['roi_labels'][:, 0, :][:, roi_scores.sum(0) >= 0].long()
        batch_dict['num_frames'] = num_frames
        roi_labels = batch_dict['roi_labels']
        batch_size = batch_dict['batch_size']
        cur_batch_boxes = copy.deepcopy(batch_dict['roi_boxes'].detach())

        batch_dict['cur_frame_idx'] = 0
        proposals_list = batch_dict['proposals_list']

        batch_dict['has_class_labels'] = True


        if self.training :
            if not self.model_cfg.get('PRE_AUG', False):
                targets_dict = self.assign_targets(batch_dict)
            else:
                targets_dict = batch_dict['targets_dict']
            batch_dict['roi_boxes'] = targets_dict['rois']
            # batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            targets_dict['trajectory_rois'][ batch_dict['cur_frame_idx'], :, :] = batch_dict['roi_boxes']
            trajectory_rois = targets_dict['trajectory_rois']
            roi_boxes = targets_dict['rois']
            empty_mask = batch_dict['roi_boxes'][:,:6].sum(-1)==0
            valid_length = targets_dict['valid_length']
            num_rois = torch.cumsum(F.pad(targets_dict['num_rois'],(1,0),'constant',0),0)
        else:
            trajectory_rois = batch_dict['trajectory_rois'][0]
            batch_dict['roi_boxes'] = trajectory_rois[ 0]
            empty_mask = batch_dict['roi_boxes'][:,:6].sum(-1)==0
            batch_dict['valid_traj_mask'] = ~empty_mask
            num_rois = torch.tensor([0,batch_dict['roi_boxes'].shape[0]],device=device)
            batch_dict['roi_labels'] = batch_dict['roi_labels'].squeeze(0)
            batch_dict['roi_scores'] = batch_dict['roi_scores'].squeeze(0)
            valid_length = batch_dict['valid_length'].squeeze(0)
        roi_boxes = batch_dict['roi_boxes']
        num_sample = self.num_lidar_points

        roi_boxes = roi_boxes.reshape(-1,roi_boxes.shape[-1])
        roi_scores = roi_scores.reshape(-1)
        roi_labels = roi_labels.reshape(-1)
        # num_rois = torch.cumsum(torch.tensor([0]+[batch_dict['roi_boxes'][i].shape[0] for i in range(batch_size)],device=device),dim=0)

        src_cur,src_idx,query_points,points_pre = self.voxel_sampler_cur(batch_size,trajectory_rois[0,...],num_sample,batch_dict,start_idx=0,num_rois=num_rois)
        batch_dict['src_cur'] = src_cur
        batch_dict['src_idx'] = src_idx
        batch_dict['query_points'] = query_points
        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK', None):
            src_cur[empty_mask.view(-1)] = 0
        src_cur = self.get_proposal_aware_geometry_feature(src_cur,trajectory_rois[0,...])
        # box_reg,box_feat = self.trajectories_auxiliary_branch(trajectory_rois.transpose(0,1),valid_length.transpose(0,1))
        hs, tokens,src_cur = self.transformer(src_cur, batch_dict, pos=None)
        if not self.training:
            key_points_root = Path('../../data/waymo/key_points_mini_new_56')/ batch_dict['metadata'][0][:-4]
            extra_key_points_root = Path('../../data/waymo/extra_key_points_mini_new') / batch_dict['metadata'][0][:-4]

            key_roi_root = Path('../../data/waymo/key_rois') / batch_dict['metadata'][0][:-4]
            src_idx = batch_dict['src_idx']
            query_points_shrink = query_points[torch.unique(src_idx)]
            os.makedirs(key_points_root,exist_ok=True)
            os.makedirs(key_roi_root,exist_ok=True)
            os.makedirs(extra_key_points_root,exist_ok=True)
            key_roi_mask = (src_idx!=0).sum(1)<28
            # np.save(key_roi_root/('%04d.npy' % batch_dict['sample_idx'][0]),torch.concat([roi_boxes[key_roi_mask],roi_scores[key_roi_mask,None],roi_labels[key_roi_mask,None].float()],dim=1).cpu().numpy())
            # np.save(key_points_root / ('%04d.npy' % batch_dict['sample_idx'][0]), query_points_shrink.cpu().numpy())
            # np.save(extra_key_points_root / ('%04d.npy' % batch_dict['sample_idx'][0]),points_pre.cpu().numpy())
            # print(self.voxel_sampler_cur.num_points/self.voxel_sampler_cur.iteration)
            if self.signal=='train':
                return batch_dict
        src_pre = self.voxel_sampler(batch_size,trajectory_rois,self.num_lidar_points2,batch_dict,num_rois)
        if False:
            src_pre = src_pre *( torch.rand(src_pre.shape, device=device) >= 0.3 )

        trajectory_rois = trajectory_rois.transpose(0,1)
        src_pre = src_pre.flatten(1,2)
        src_idx = batch_dict['src_idx'][:,:self.num_lidar_points2]
        src_pre[:,:self.num_lidar_points2] = torch.gather(query_points,0,src_idx.reshape(-1,1).repeat(1,query_points.shape[-1])).unflatten(0,src_idx.shape)
        if False:
            drop_indice = torch.ones()

        src_pre = self.get_proposal_aware_motion_feature(src_pre, trajectory_rois,valid_length)


        tokens2 = self.transformer2st(src_pre,tokens[-1],src_cur)

        point_cls_list = []
        point_reg_list = []

        for i in range(len(tokens)):
            point_cls_list.append(self.class_embed[0](tokens[i][:,0]))
        for i in range(len(tokens2)):
            point_cls_list.append(self.class_embed_final(tokens2[i][:,-1]))


        for j in range(len(tokens)):
            point_reg_list.append(self.bbox_embed[0](tokens[j][:,0]))
        # for j in range(len(tokens2)):
        #     for k in range(tokens2[0].shape[1]):
        #         point_reg_list.append(self.bbox_embed[k](tokens2[j][:,k]))

        point_cls = torch.cat(point_cls_list,0)

        point_reg = torch.cat(point_reg_list,0)


        joint_reg = self.jointembed(tokens2[-1].flatten(1,2))



        rcnn_cls = point_cls
        rcnn_reg = joint_reg

        if not self.training:
            rcnn_cls = rcnn_cls[-tokens2[-1].shape[0]:]

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['roi_boxes'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            batch_dict['cls_preds_normalized'] = False
            if self.avg_stage1_score:
                stage1_score = batch_dict['roi_scores'][:,None]
                batch_cls_preds = F.sigmoid(batch_cls_preds)
                if self.model_cfg.get('IOU_WEIGHT', None):
                    batch_box_preds_list = []
                    roi_labels_list = []
                    batch_cls_preds_list = []
                    # for bs_idx in range(batch_size):

                    car_mask = batch_dict['roi_labels'] ==1
                    batch_cls_preds_car = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[0])* \
                                          stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[0])
                    batch_cls_preds_car = batch_cls_preds_car[car_mask][None]
                    batch_cls_preds_pedcyc = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[1])* \
                                             stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[1])
                    batch_cls_preds_pedcyc = batch_cls_preds_pedcyc[~car_mask][None]
                    cls_preds = torch.cat([batch_cls_preds_car,batch_cls_preds_pedcyc],1)
                    box_preds = torch.cat([batch_dict['batch_box_preds'][car_mask],
                                                 batch_dict['batch_box_preds'][~car_mask]],0)[None]
                    roi_labels = torch.cat([batch_dict['roi_labels'][car_mask],
                                            batch_dict['roi_labels'][~car_mask]],0)[None]
                    batch_box_preds_list.append(box_preds)
                    roi_labels_list.append(roi_labels)
                    batch_cls_preds_list.append(cls_preds)


                    batch_dict['batch_box_preds'] = torch.cat(batch_box_preds_list,0)
                    batch_dict['roi_labels'] = torch.cat(roi_labels_list,0)
                    batch_cls_preds = torch.cat(batch_cls_preds_list,0)
                    
                else:
                    batch_cls_preds = torch.sqrt(batch_cls_preds*stage1_score)
                batch_dict['cls_preds_normalized']  = True

            batch_dict['batch_cls_preds'] = batch_cls_preds
        else:
            targets_dict['batch_size'] = batch_size
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['box_reg'] = rcnn_reg
            targets_dict['point_reg'] = point_reg
            targets_dict['point_cls'] = point_cls
            self.forward_ret_dict = targets_dict
        # self.delay.append(time.time()-st)
        # import pycuda.driver as cuda
        # cuda.init()
        # device_0 = cuda.Device(0)
        # self.memory_num.append((device_0.total_memory() - cuda.mem_get_info()[0]) / 1024 / 1024)
        # torch.cuda.empty_cache()
        # print(sum(self.delay)/len(self.delay))
        # print(sum(self.memory_num)/len(self.delay))
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

        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)

        rcnn_reg = forward_ret_dict['rcnn_reg'] 

        roi_boxes3d = forward_ret_dict['rois']

        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}
        
        
        if loss_cfgs.REG_LOSS == 'smooth-l1':

            rois_anchor = roi_boxes3d.clone().detach()[:,:7].contiguous().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
                )
            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
  
            if self.model_cfg.USE_AUX_LOSS:
                point_reg = forward_ret_dict['point_reg']

                groups = point_reg.shape[0]//reg_targets.shape[0]
                if groups != 1 :
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                        point_reg[i*slice:(i+1)*slice].view(slice, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),) 
                        point_loss_reg = (point_loss_reg.view(slice, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                        point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]
                        
                        point_loss_regs += point_loss_reg
                    point_loss_regs = point_loss_regs / groups
                    tb_dict['point_loss_reg'] = point_loss_regs.item()
                    rcnn_loss_reg += point_loss_regs 

                else:
                    point_loss_reg = self.reg_loss_func(point_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)  
                    point_loss_reg = (point_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                    point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]
                    tb_dict['point_loss_reg'] = point_loss_reg.item()
                    rcnn_loss_reg += point_loss_reg

                seqbox_reg = forward_ret_dict['box_reg']  
                seqbox_loss_reg = self.reg_loss_func(seqbox_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)
                seqbox_loss_reg = (seqbox_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                seqbox_loss_reg = seqbox_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][1]
                tb_dict['seqbox_loss_reg'] = seqbox_loss_reg.item()
                rcnn_loss_reg += seqbox_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:

                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d[:,:7].contiguous().view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view( -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:,  6].view(-1)
                roi_xyz = fg_roi_boxes3d[:,  0:3].view(-1, 3)
                batch_anchors[:,  0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], code_size), batch_anchors
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
                    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat[i*slice:(i+1)*slice]), 
                                     rcnn_cls_labels.float(), reduction='none')

                    cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                    rcnn_loss_cls = rcnn_loss_cls + (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

                rcnn_loss_cls = rcnn_loss_cls / groups


            else:
                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
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
            batch_cls_preds = cls_preds.view( -1, cls_preds.shape[-1])
        else:
            batch_cls_preds = None
        batch_box_preds = box_preds.view( -1, code_size)

        roi_ry = rois[:,  6].view(-1)
        roi_xyz = rois[:,  0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:,  0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view( -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds,rois[:,7:]],-1)
        return batch_cls_preds, batch_box_preds
