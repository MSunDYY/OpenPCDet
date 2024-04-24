from os import getgrouplist
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from spconv.pytorch.utils import PointToVoxel
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import ball_query, grouping_operation, QueryAndGroup
from pcdet import device
from pcdet.ops.box2map.box2map import points2box_gpu,points2box
from pcdet.ops.iou3d_nms import iou3d_nms_utils

class PointNetfeat(nn.Module):
    def __init__(self, input_dim, x=1, outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel == 256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(input_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(256 * x)
        self.bn4 = nn.BatchNorm1d(self.output_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x))

        x = torch.max(x_ori, 2, keepdim=True)[0]

        x = x.view(-1, self.output_channel)
        return x, x_ori


class PointNet2(nn.Module):
    def __init__(self, input_dim, model_cfg=None):
        super(PointNet2, self).__init__()
        num_dim = model_cfg.POINT_DIM
        self.sampler = QueryAndGroup(radius=1, nsample=4, use_xyz=False)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(num_dim)
        self.fc1 = nn.Linear(num_dim, num_dim)
        self.fc_ce = nn.Linear(num_dim, 3)
        self.fc2 = nn.Linear(num_dim, num_dim)
        self.fc_lwh = nn.Linear(num_dim, 3)
        self.fc3 = nn.Linear(num_dim, num_dim)
        self.fc_theta = nn.Linear(num_dim, 1)
        self.out_dim = 3 + 3 + 1

    def forward(self, x):
        num_batch_cnt = (torch.ones(x.shape[0]) * x.shape[1] * x.shape[2]).to(device)
        num_batch_new_cnt = (torch.ones(x.shape[0]) * x.shape[2]).to(device)
        B = x.shape[0]
        new_xyz = x[:, 0, :, :3].reshape(-1, 3)
        xyz = x[:, 1:, :, :3].reshape(-1, 3)

        x, _ = self.sampler(xyz.contiguous(), num_batch_cnt.contiguous().int(), new_xyz.contiguous(),
                            num_batch_new_cnt.contiguous().int(), x.reshape(-1, x.shape[-1]))
        # x = x.reshape(B,-1,x.shape[-2],x.shape[-1])
        x = self.bn1(self.conv1(x))
        x = torch.max(x, dim=-1)[0]
        feature = x.reshape(B, -1, x.shape[-1])
        x1 = F.relu(self.bn1(self.fc1(x)))
        x_center = self.fc_ce(x1)
        x2 = F.relu(self.bn1(self.fc2(x)))
        x_lwh = self.fc_lwh(x2)
        x3 = F.relu(self.bn1(self.fc3(x)))
        x_theta = self.fc_theta(x3)

        return torch.concat((x_center, x_lwh, x_theta), dim=-1).reshape(B, -1, self.out_dim), feature


class PointNet(nn.Module):
    def __init__(self, input_dim, joint_feat=False, model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT

        times = 1
        self.feat = PointNetfeat(input_dim, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, channels)

        self.pre_bn = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.fc_s1 = nn.Linear(channels * times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_ce1 = nn.Linear(channels * times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(channels * times, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)

    def forward(self, x, feat=None):

        if self.joint_feat:
            if len(feat.shape) > 2:
                feat = torch.max(feat, 2, keepdim=True)[0]
                x = feat.view(-1, self.output_channel)
                x = F.relu(self.bn1(self.fc1(x)))
                feat = F.relu(self.bn2(self.fc2(x)))
            else:
                feat = feat
            feat_traj = None
        else:
            x, feat_traj = self.feat(self.pre_bn(x))
            x = F.relu(self.bn1(self.fc1(x)))
            feat = F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return torch.cat([centers, sizes, headings], -1), feat, feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SpatialMixerBlock(nn.Module):

    def __init__(self, hidden_dim, grid_size, channels, config=None, dropout=0.0):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
        self.config = config
        self.grid_size = grid_size

    def forward(self, src):
        src2 = self.mixer(src, src, src)[0]

        src = src + self.dropout(src2)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)

        return src_mixer


class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False,
                 num_lidar_points=None, share_head=True, num_groups=None,
                 sequence_stride=None, num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head

        self.nhead = nhead
        self.sequence_stride = sequence_stride

        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward, dropout, activation,
                                                 normalize_before, num_lidar_points, num_groups=2) for i in
                         range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, self.config)

        self.token = nn.Parameter(torch.zeros(1, 1, d_model))

        # if self.num_frames > 4:
        #     self.group_length = self.num_frames // self.num_groups
        #     self.fusion_all_group = MLP(input_dim=self.config.hidden_dim * self.group_length,
        #                                 hidden_dim=self.config.hidden_dim, output_dim=self.config.hidden_dim,
        #                                 num_layers=4)
        #
        #     self.fusion_norm = FFN(d_model, dim_feedforward)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,num_groups,batch_size,num_frames, pos=None):

        # BS, F,N,K, C = src.shape
        if not pos is None:
            pos = pos.permute(1, 0, 2)

        # if self.num_frames == 16:
        #     token_list = [self.token[i:(i + 1)].repeat(BS, 1, 1) for i in range(self.num_groups)]
        #     if self.sequence_stride == 1:
        #         src_groups = src.view(src.shape[0], src.shape[1] // self.num_groups, -1).chunk(4, dim=1)
        #
        #     elif self.sequence_stride == 4:
        #         src_groups = []
        #
        #         for i in range(self.num_groups):
        #             groups = []
        #             for j in range(self.group_length):
        #                 points_index_start = (i + j * self.sequence_stride) * self.num_proxy_points
        #                 points_index_end = points_index_start + self.num_proxy_points
        #                 groups.append(src[:, points_index_start:points_index_end])
        #
        #             groups = torch.cat(groups, -1)
        #             src_groups.append(groups)
        #
        #     else:
        #         raise NotImplementedError
        #
        #     src_merge = torch.cat(src_groups, 1)
        #     src = self.fusion_norm(src[:, :self.num_groups * self.num_proxy_points], self.fusion_all_group(src_merge))
        #     src = [torch.cat([token_list[i], src[:, i * self.num_proxy_points:(i + 1) * self.num_proxy_points]], dim=1)
        #            for i in range(self.num_groups)]
        #     src = torch.cat(src, dim=0)
        #
        # else:
        token_list = self.token.to(device)
        # src = src.permute(1,0,2)
        
        xyz_vel = src[:,:,-5:]
        
        src = torch.cat([token_list.repeat(1, src.shape[1], 1), src[:,:,:-5]], dim=0)

        # src = src.permute(1, 0, 2)
        src,weight = self.encoder(src, pos=pos)
        token = [src[:1][:,num_groups[i*num_frames]:num_groups[i*num_frames+1]] for i in range(batch_size)]
        src = torch.concat([src[1:],xyz_vel],dim=-1)
        
        # memory = torch.cat(memory[0:1].chunk(4, 1), 0)
        return src,torch.concat(token,1),weight


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,
                pos: Optional[Tensor] = None):

        token_list = []
        output = src
        for layer in self.layers:
            output,weight = layer(output, pos=pos)
            # token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,weight
class vector_attention(nn.Module):
    def __init__(self,d_model,nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.linear_q = nn.ModuleList()
        self.linear_w = nn.ModuleList()
        self.linear_k = nn.ModuleList()
        self.linear_v = nn.ModuleList()
        for i in range(nhead):

            self.linear_w.append(nn.Sequential(nn.BatchNorm1d(d_model),nn.ReLU(inplace=True),
                                          nn.Linear(d_model,d_model),
                                          nn.BatchNorm1d(d_model),nn.ReLU(inplace=True),
                                          nn.Linear(d_model,d_model)))
            self.softmax = nn.Softmax(dim=0)
            self.linear_q.append(nn.Linear(d_model,d_model))
            self.linear_k.append(nn.Linear(d_model, d_model))
            self.linear_v.append(nn.Linear(d_model, d_model))
            self.MLP = MLP(d_model*nhead,hidden_dim=d_model,output_dim=d_model,num_layers=3)
    def forward(self,q,k,v,pos=None):
        x_list = []
        w_all=0
        for i in range(self.nhead):
            q,k,v = self.linear_q[i](q),self.linear_k[i](k),self.linear_v[i](v)
            w = k -q
            for j,layer in enumerate(self.linear_w[i]): w =layer(w.permute(1,2,0).contiguous()).permute(2,0,1).contiguous() if j%3==0 else layer(w)
            w = self.softmax(w)
            x = (w*v).sum(0).unsqueeze(0)
            x_list.append(x)
            w_all = w.sum(-1) if i==0 else w_all+w.sum(-1)
        # w_all = w_all/w_all.sum(0)[None,:]
        x =self.MLP(torch.concat(x_list,dim=-1))
        return x,w_all




class TransformerEncoderLayer(nn.Module):
    count = 0

    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_points=None, num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
        self.config = config
        self.num_point = num_points
        self.num_groups = num_groups
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = vector_attention(d_model,nhead = 2 )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.layer_count <= self.config.enc_layers - 1:
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(self.num_groups):
                self.cross_attn_layers.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))

            self.ffn = FFN(d_model, dim_feedforward)
            self.fusion_all_groups = MLP(input_dim=d_model * 4, hidden_dim=d_model, output_dim=d_model, num_layers=4)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,
                                              self.config.use_mlp_mixer.get('grid_size', 4), self.config.hidden_dim,
                                              self.config.use_mlp_mixer)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):

        src_intra_group_fusion = self.mlp_mixer_3d(src[1:])
        src = torch.cat([src[:1], src_intra_group_fusion], 0)

        token = src[:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion

        src_summary,weight = self.self_attn(token, key, src_intra_group_fusion)

        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = torch.cat([token, src[1:]], 0)

        if self.layer_count <= self.config.enc_layers - 1:

            src_all_groups = src[1:].view((src.shape[0] - 1) * 4, -1, src.shape[-1])
            src_groups_list = src_all_groups.chunk(self.num_groups, 0)

            src_all_groups = torch.cat(src_groups_list, -1)
            src_all_groups_fusion = self.fusion_all_groups(src_all_groups)

            key = self.with_pos_embed(src_all_groups_fusion, pos[1:])
            query_list = [self.with_pos_embed(query, pos[1:]) for query in src_groups_list]

            inter_group_fusion_list = []
            for i in range(self.num_groups):
                inter_group_fusion = self.cross_attn_layers[i](query_list[i], key, value=src_all_groups_fusion)[0]
                inter_group_fusion = self.ffn(src_groups_list[i], inter_group_fusion)
                inter_group_fusion_list.append(inter_group_fusion)

            src_inter_group_fusion = torch.cat(inter_group_fusion_list, 1)

            src = torch.cat([src[:1], src_inter_group_fusion], 0)
        
        return src,weight

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, dout=None,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def build_transformer(args,num_groups):
    return nn.ModuleList([Transformer(
        config=args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        num_lidar_points=args.num_lidar_points[i],
        sequence_stride=args.get('sequence_stride', 1),

    ) for i in range(num_groups)])


class VoxelSampler(nn.Module):
    GAMMA = 1.1

    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5,training = True):
        super().__init__()

        self.voxel_size = voxel_size

        self.gen = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, pc_range[5] - pc_range[2]],
            coors_range_xyz=pc_range,
            num_point_features=num_point_features,
            max_num_voxels=50000,
            max_num_points_per_voxel=max_points_per_voxel,
            device=device
        )
        self.training = training
        self.pc_start = torch.FloatTensor(pc_range[:2]).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)

    def get_output_feature_dim(self):
        return self.num_point_features

    def cylindrical_pool(self, cur_points, cur_boxes, num_sample, gamma=1.,pool='even',next_boxes = None,next_idx=0):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])
        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        if pool=='even':

            sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample)
            rois_new = cur_boxes
            roi_mask = cur_boxes.new_ones(cur_boxes.shape[0],dtype = torch.bool)
        else:
            sampled_mask, sampled_idx = self.select_points(point_mask=point_mask.int(), num_sampled_per_box=num_sample,
                                                       num_sampled_per_point=2)
            roi_mask = sampled_mask.sum(-1) > 0

                
            rois_new = cur_boxes[roi_mask]
            sampled_idx = sampled_idx[roi_mask]
            sampled_mask = sampled_mask[roi_mask]
        # sampled_point_mask[torch.arange(point_mask.shape[0]).unsqueeze(1),sampled_idx] = 1
        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx.long()).view(len(rois_new), num_sample, cur_points.shape[-1])
        sampled_points = torch.concat([sampled_points,rois_new[:,None,-2:].repeat(1,num_sample,1)],dim=-1)
        sampled_points[sampled_mask == 0, :] = 0
        if next_boxes is not None:
            next_boxes = next_boxes[next_boxes[:,:3].sum(-1)!=0]
            cur_boxes = cur_boxes.clone()
            cur_boxes[:,:2]=cur_boxes[:,:2]-cur_boxes[:,-2:]*next_idx
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_boxes[:,:7],next_boxes[:,:7])

            roi_mask = iou3d.sum(-1)>0
            if roi_mask.sum().item()==0:
                roi_mask[:] = True
            # next_radiis = torch.norm(next_boxes[:, 3:5] / 2, dim=-1) * gamma**next_idx
            # sampled_points = sampled_points.reshape(-1,sampled_points.shape[-1])
            # dis = torch.norm(
            #     ((sampled_points[:, :2]-sampled_points[:,-2:]*next_idx).unsqueeze(1) - next_boxes[:, :2].unsqueeze(0).repeat(sampled_points.shape[0],1, 1)),
            #     dim=2)
            # point_mask = (dis <= next_radiis.unsqueeze(0))
            # point_mask = point_mask.reshape(-1,num_sample,dis.shape[-1])
            # cur_mask = (point_mask.reshape(point_mask.shape[0],-1).sum(-1)>1)
            # sampled_points = sampled_points.reshape(cur_mask.shape[0],num_sample,sampled_points.shape[-1])
            # temp = roi_mask.clone()
            # roi_mask[temp] = temp[temp]*cur_mask
            return sampled_points[roi_mask], rois_new[roi_mask], roi_mask
        return sampled_points, rois_new,roi_mask

    def select_points(self, point_mask, num_sampled_per_box, num_sampled_per_point=2):

        sampled_mask = point_mask.new_zeros(point_mask.shape[0], num_sampled_per_box,device='cpu')
        sampled_idx = point_mask.new_zeros(point_mask.shape[0], num_sampled_per_box,device = 'cpu')
        point_sampled_num = point_mask.new_zeros(point_mask.shape[0],device='cpu').int()
        points2box(point_mask.to('cpu').contiguous(), sampled_mask, sampled_idx, point_sampled_num, num_sampled_per_box,
                       num_sampled_per_point)

        # point_mask_cpu = point_mask.to('cpu')
        # sampled_mask_cpu = point_mask_cpu.new_zeros(point_mask.shape[0], num_sampled_per_box)
        # sampled_idx_cpu = point_mask_cpu.new_zeros(point_mask.shape[0], num_sampled_per_box)
        # point_sampled_num_cpu = point_mask_cpu.new_zeros(point_mask.shape[0]).int()
        # points2box(point_mask_cpu.contiguous(), sampled_mask_cpu, sampled_idx_cpu, point_sampled_num_cpu, num_sampled_per_box,
        #                num_sampled_per_point-1)
        return sampled_mask.to(device), sampled_idx.to(device)

    def forward(self, batch_size, trajectory_rois, num_sample, batch_dict):

        src = list()
        rois = list()
        roi_scores = list()

        roi_labels = list()
        trajectory_rois = batch_dict['roi_boxes']
        trajectory_scores = batch_dict['roi_scores']
        trajectory_labels = batch_dict['roi_labels']
        for bs_idx in range(batch_size):

            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]
            # cur_batch_boxes = trajectory_rois[bs_idx]
            src_single_batch = list()
            rois_single_batch = list()
            roi_labels_single_batch = list()
            roi_scores_single_batch = list()
            for idx in range(trajectory_rois.shape[1]):
                gamma = self.GAMMA  # ** (idx+1)
                
                time_mask = (cur_points[:, -1] - idx * 0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask, :].contiguous()
                if self.training and idx==0:
                    cur_frame_boxes = batch_dict['rois_cur'][bs_idx]
                    cur_frame_scores = batch_dict['roi_scores_cur'][bs_idx]
                    cur_frame_labels = batch_dict['roi_labels_cur'][bs_idx]
                else:
                    mask = trajectory_scores[bs_idx,idx]!=0
                    cur_frame_boxes = trajectory_rois[bs_idx,idx][mask]
                    cur_frame_scores = trajectory_scores[bs_idx,idx][mask]
                    cur_frame_labels = trajectory_labels[bs_idx,idx][mask]

                voxel, coords, num_points = self.gen(cur_time_points)
                coords = coords[:, [2, 1]].contiguous()

                query_coords = (cur_frame_boxes[:, :2] - self.pc_start) // self.voxel_size

                radiis = torch.ceil(
                    torch.norm(cur_frame_boxes[:, 3:5] / 2, dim=-1) * gamma / self.voxel_size)

                # h_table = torch.zeros(self.grid_x*self.grid_y).fill_(-1).to(coords)
                # coords_ = coords[:, 0] * self.grid_y + coords[:, 1]
                # h_table[coords_.long()] = torch.arange(len(coords)).to(coords)

                # v_indice = torch.zeros((len(query_coords)), int(radiis.max())**2).fill_(-1).to(coords)
                # scatter.hash_query(self.grid_x, self.grid_y, query_coords.int(), radiis.int(), h_table, v_indice)
                # v_indice = v_indice.long()

                # voxel_points = voxel[v_indice, :, :]
                # num_points = num_points[v_indice, None]

                # cur_radiis = torch.norm(cur_frame_boxes[:, None, None, 3:5]/2, dim=-1) * gamma
                # dis = torch.norm(voxel_points[:, :, :, :2] - cur_frame_boxes[:, None, None, :2], dim = -1)
                # point_mask = dis <= cur_radiis

                # a, b, _ = num_points.shape
                # points_mask = point_mask & (v_indice[:, :, None]!=-1) & \
                #     (num_points > torch.arange(self.k)[None, None, :].repeat(a, b, 1).type_as(num_points))

                # points_mask = points_mask.flatten(1, 2)
                # voxel_points = voxel_points.flatten(1, 2)

                # random_perm = torch.randperm(points_mask.shape[1])
                # points_mask = points_mask[:, random_perm]
                # voxel_points = voxel_points[:, random_perm, :]

                # try:
                #     sampled_mask, sampled_idx = torch.topk(points_mask.float(), num_sample)

                #     key_points = torch.gather(voxel_points, 1, sampled_idx[:, :, None].repeat(1, 1, voxel_points.shape[-1]))
                #     key_points[sampled_mask==0, :] = 0
                # except:
                #     key_points = torch.zeros([len(cur_frame_boxes), num_sample, 5]).to(voxel)
                #     key_points[:, :voxel_points.shape[1], :] = voxel_points

                dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :])

                voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)

                num_points = num_points[voxel_mask]
                key_points = voxel[voxel_mask, :]

                point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

                point_mask = num_points[:, None] > point_mask
                key_points = key_points[point_mask]
                key_points = key_points[torch.randperm(len(key_points)), :]
                assign_list = [0,0,0,2,0,4,4,6]
                key_points, rois_new,roi_mask = self.cylindrical_pool(key_points, cur_frame_boxes, num_sample, gamma,pool='even',next_boxes=None if idx==0 else rois_single_batch[assign_list[idx]],next_idx=idx-assign_list[idx])

                src_single_batch.append(key_points)
                rois_single_batch.append(rois_new)
                roi_scores_single_batch.append(cur_frame_scores[roi_mask])
                roi_labels_single_batch.append(cur_frame_labels[roi_mask])
            src+=src_single_batch
            rois+=rois_single_batch
            roi_scores+=roi_scores_single_batch
            roi_labels_single_batch+=roi_labels_single_batch
        batch_dict['src_list'] = src
        batch_dict['rois_list'] = rois
        batch_dict['roi_scores_list'] = roi_scores
        batch_dict['roi_labels_list'] = roi_labels
        return batch_dict


def build_voxel_sampler(device,training=True):
    return VoxelSampler(
        device,
        voxel_size=0.4,
        pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
        max_points_per_voxel=32,
        num_point_features=6,
        training=training
    )
