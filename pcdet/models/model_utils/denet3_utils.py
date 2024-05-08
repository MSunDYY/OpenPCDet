from os import getgrouplist
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from spconv.pytorch.utils import PointToVoxel

# from torch.utils.cpp_extension import load
# scatter = load(name='scatter', sources=['../cuda/scatter.cpp', '../cuda/scatter.cu'])
class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, grid_size, channels, config=None, dropout=0.0):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.pos_linear = nn.Linear(3,channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
        self.config = config
        self.grid_size = grid_size

    def forward(self, src1,src2,xyz1,xyz2):
        xyz1 = self.pos_linear(xyz1).permute(1,0,2)
        xyz2 = self.pos_linear(xyz2).permute(1,0,2)
        src1 = src1.permute(1,0,2)
        src2 = src2.permute(1,0,2)
        src = self.mixer(src1+xyz1, src2+xyz2, src2)[0]

        src = src1 + self.dropout(src)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)

        return src_mixer.permute(1,0,2)

class BioCrossAttention(nn.Module):

    def __init__(self, hidden_dim, grid_size, channels, config=None, dropout=0.0):
        super().__init__()

        self.mixer1 = nn.MultiheadAttention(channels, 4, dropout=dropout)
        self.mixer2 = nn.MultiheadAttention(channels, 4, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.pos_linear = nn.Linear(3,channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn1 = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )

        self.config = config
        self.grid_size = grid_size

    def forward(self, src1,src2,xyz1,xyz2,return_xyz=False):
        xyz1_emb = xyz1
        xyz2_emb = xyz2
        # src1 = src1.permute(1,0,2)
        # src2 = src2.permute(1,0,2)
        src1_= self.mixer1(src1+xyz1_emb, src2+xyz2_emb, src2)[0]
        src2_ = self.mixer2(src2+xyz2_emb,src1+xyz1_emb,src1)[0]
        src1 = src1 + self.dropout(src1_)
        src2 = src2 + self.dropout(src2_)
        src_mixer1 = self.norm(src1)
        src_mixer2 = self.norm(src2)

        src_mixer1 = src_mixer1 + self.ffn1(src_mixer1)
        src_mixer2 = src_mixer2 + self.ffn2(src_mixer2)
        src_mixer1 = self.norm_channel(src_mixer1)
        src_mixer2 = self.norm_channel(src_mixer2)
        if return_xyz:
            return torch.concat([src_mixer1,xyz1],dim=-1),torch.concat([src_mixer2,xyz2],dim=-1)
        else:
            return src_mixer1,src_mixer2
class PointNetfeat(nn.Module):
    def __init__(self, input_dim, x=1,outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel==256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(input_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x,  self.output_channel, 1)
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

class PointNet(nn.Module):
    def __init__(self, input_dim, joint_feat=False,model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT

        times=1
        self.feat = PointNetfeat(input_dim, 1)

        self.fc1 = nn.Linear(512, 256 )
        self.fc2 = nn.Linear(256, channels)

        self.pre_bn = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.fc_s1 = nn.Linear(channels*times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_ce1 = nn.Linear(channels*times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(channels*times, 256)
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

        return torch.cat([centers, sizes, headings],-1),feat,feat_traj

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

    def __init__(self, hidden_dim, grid_size,channels,config=None, dropout=0.0):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
                               nn.Linear(channels, 2*channels),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(2*channels, channels),
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
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                num_lidar_points=None, share_head=True,num_groups=None,
                sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups
        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward,dropout, activation, 
                      normalize_before, num_lidar_points,num_groups=num_groups) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)

        self.token = nn.Parameter(torch.zeros(self.num_groups*2, 1, d_model))
        self.token2 = nn.Parameter(torch.zeros(self.num_groups,1,d_model))
        
        if self.num_frames >4:
  
            self.group_length = self.num_frames // self.num_groups
            self.fusion_all_group = MLP(input_dim = self.config.hidden_dim*self.group_length, 
               hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos=None):

        BS, N, C = src.shape
        # if not pos is None:
        #     pos = pos.permute(1, 0, 2)
            
        
        token_list = [self.token[i:(i+1)].repeat(BS,1,1) for i in range(self.num_groups)]
        src = [torch.cat([token_list[i],src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        src = torch.cat(src,dim=0)
        pos = [pos[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points] for i in range(self.num_groups)]
        pos = torch.cat(pos,dim=0)
        # src2 = [torch.cat([token_list[i],src2[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        # src2 = torch.cat(src2,dim=0)
        src = src.permute(1, 0, 2)
        pos = pos.permute(1, 0, 2)
        # src2 = src2.permute(1,0,2)
        memory,tokens = self.encoder(src,pos=pos)

        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
        return memory, tokens
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
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
            output,tokens = layer(output,pos=pos)
            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,token_list
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

        for i in range(self.nhead):
            q,k,v = self.linear_q[i](q),self.linear_k[i](k),self.linear_v[i](v)
            w = k -q
            for j,layer in enumerate(self.linear_w[i]): w =layer(w.permute(1,2,0).contiguous()).permute(2,0,1).contiguous() if j%3==0 else layer(w)
            w = self.softmax(w)
            x = (w*v).sum(0).unsqueeze(0)
            x_list.append(x)
        # w_all = w_all/w_all.sum(0)[None,:]
        x =self.MLP(torch.concat(x_list,dim=-1))
        return x

class TransformerEncoderLayer(nn.Module):
    count = 0
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
        self.config = config
        self.num_point = num_points
        self.num_groups= num_groups
        # self.self_attn = vector_attention(d_model, nhead=1)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.selfattn2 = nn.MultiheadAttention(d_model,nhead,dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        if self.layer_count <= self.config.enc_layers-1:
            self.cross_conv_1 = nn.Linear(d_model * 2, d_model)
            self.cross_norm_1 = nn.LayerNorm(d_model)
            self.cross_conv_2 = nn.Linear(d_model * 2, d_model)
            self.cross_norm_2 = nn.LayerNorm(d_model)


        self.bio_cross_atten = BioCrossAttention(3,4,self.config.hidden_dim,None)
        self.cross_atten = CrossAttention(3,4,self.config.hidden_dim,None)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(
                        self.config.use_mlp_mixer.hidden_dim, 
                        self.config.use_mlp_mixer.get('grid_size', 4), 
                        self.config.hidden_dim, 
                        self.config.use_mlp_mixer
        )
        self.mlp_mixer_3d2 = SpatialMixerBlock(
                        self.config.use_mlp_mixer.hidden_dim,
                        self.config.use_mlp_mixer.get('grid_size', 4),
                        self.config.hidden_dim,
                        self.config.use_mlp_mixer
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):
        # src = torch.cat([src1,src2],0)
        num_rois = src.shape[1]//(self.num_groups*2)
        # src1,src2 = src.chunk(2,1)

        src_intra_group_fusion = self.mlp_mixer_3d(src[1:])
        # src_intra_group_fusion2 = self.mlp_mixer_3d2(src2[1:])
        # src_intra_group_fusion2 = self.mlp_mixer_3d(src2[1:])
        src = torch.cat([src[:1],src_intra_group_fusion],0)
        # src2 = torch.cat([src2[:1],src_intra_group_fusion2],0)
        token = src[:1]
        # token2 = src2[:1]
        # token2 = src2[:1]
        # if not pos is None:
        #     key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        # else:
        key = src_intra_group_fusion

        src_summary = self.self_attn(token, key,src_intra_group_fusion)[0]
        # src_summary2 = self.self_attn2(token2, key2,src_intra_group_fusion2)[0]

        token = token + self.dropout1(src_summary)
        token = self.norm(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm(token)
        src = torch.cat([token,src[1:]],0)

        if self.layer_count <= self.config.enc_layers-1:
            src_max = torch.max(src[1:],dim=0,keepdim=True)[0].repeat(src.shape[0]-1,1,1)

            src=torch.concat([src[:1],self.cross_norm_1(src[1:].clone()+self.cross_conv_1(torch.concat([src[1:],src_max],-1)))],0)
            src_all_groups = torch.concat([src[1:],pos],-1)
            # num_half = pos.shape[1]//2
            # src_all_groups = torch.stack(src_all_groups.chunk(2,1))
            src_all_groups = src_all_groups.chunk(self.num_groups,1)
            if self.layer_count==1:
                # src1 = torch.concat([src_all_groups[0],src_all_groups[2],src_all_groups[4],src_all_groups[6]],dim=1)
                # src2 = torch.concat([src_all_groups[1],src_all_groups[3],src_all_groups[5],src_all_groups[7]],dim=1)
                src1 = torch.concat([src_all_groups[0], src_all_groups[2]], dim=1)
                src2 = torch.concat([src_all_groups[1], src_all_groups[3]], dim=1)
            else:
                # src1 = torch.concat([src_all_groups[0], src_all_groups[1],src_all_groups[4],src_all_groups[5]], dim=1)
                # src2 = torch.concat([src_all_groups[2], src_all_groups[3],src_all_groups[6],src_all_groups[7]], dim=1)
                src1 = torch.concat([src_all_groups[0], src_all_groups[1]], dim=1)
                src2 = torch.concat([src_all_groups[2], src_all_groups[3]], dim=1)

            # src2 = src_all_groups[1:,:num_half,:]
            # src_cur = torch.concat([src1[:,:num_rois*(self.num_groups-1)],src2[:,:num_rois*(self.num_groups-1)]],dim=1)
            # src_pre = torch.concat([src1[:,num_rois:],src2[:,num_rois:]],dim=1)
            # src_cur[:,:,-5:-3]+=src_cur[:,:,-2:]


            src_1,src_2 = self.bio_cross_atten(src1[:,:,:-self.config.hidden_dim],src2[:,:,:-self.config.hidden_dim],src1[:,:,-self.config.hidden_dim:],src2[:,:,-self.config.hidden_dim:])
            src_all_groups = src_1.chunk(self.num_groups//2,1) + src_2.chunk(self.num_groups//2,1)
            if self.layer_count==1:
                # indice = [0,4,1,5,2,6,3,7]
                indice = [0,2,1,3]
            else:
                # indice = [0,1,5,6,2,3,6,7]
                indice = [0,1,2,3]
            src = torch.concat([src_all_groups[i] for i in indice],1)
            src = torch.concat([token,src],dim=0)
        # if self.layer_count <= self.config.enc_layers-1:
        #
        #     # src1 = src[:,:src.shape[1]//2]
        #     # src2 = src[:,src.shape[1]//2:]
        #     num_half = pos.shape[1]//2
        #     num_points = src.shape[0]-1
        #     src_all_groups = src[1:].view((src.shape[0]-1)*8,-1,src.shape[-1])
        #     # pos = pos.reshape(pos.shape[0]*8,-1,pos.shape[-1])
        #     # src_groups_list = src_all_groups.chunk(self.num_groups,0)
        #     # src_groups_list = [src_all_groups[torch.arange(num_points)*8+i] for i in range(8)]
        #     pos1 = pos[:,:num_half]
        #     pos2 = pos[:,num_half:]
        #     src_all_groups1 = src[1:,:num_half]
        #     src_all_groups2 = src[1:,num_half:]
        #     src_all_groups1,src_all_groups2 = self.cross_atten(src_all_groups1,src_all_groups2,pos1,pos2)
        #
        #     src_all_groups1 = src_all_groups1.view((src_all_groups1.shape[0])*4,-1,src.shape[-1])
        #     src_groups_list1 = [src_all_groups1[torch.arange(num_points)*4+i] for i in range(4)]
        #     src_all_groups1 = torch.stack(src_groups_list1)
        #     src_max_groups1 = torch.max(src_all_groups1, 1, keepdim=True).values
        #     src_past_groups1 =  torch.cat([src_all_groups1[1:],\
        #          src_max_groups1[:-1].repeat(1, (src.shape[0]-1), 1, 1)], -1)
        #     src_all_groups1[1:] = self.cross_norm_1(self.cross_conv_1(src_past_groups1) + src_all_groups1[1:])
        #
        #     src_max_groups1 = torch.max(src_all_groups1, 1, keepdim=True).values
        #     src_past_groups1 =  torch.cat([src_all_groups1[:-1],\
        #          src_max_groups1[1:].repeat(1, (src.shape[0]-1), 1, 1)], -1)
        #     src_all_groups1[:-1] = self.cross_norm_2(self.cross_conv_2(src_past_groups1) + src_all_groups1[:-1])
        #
        #     src_inter_group_fusion1 = src_all_groups1.permute(1, 0, 2, 3).contiguous().flatten(1,2)
        #
        #     src_all_groups2 = src_all_groups2.view((src_all_groups2.shape[0]) * 4, -1, src.shape[-1])
        #     src_groups_list2 = [src_all_groups2[torch.arange(num_points) * 4 + i] for i in range(4)]
        #     src_all_groups2 = torch.stack(src_groups_list2)
        #     src_max_groups2 = torch.max(src_all_groups2, 1, keepdim=True).values
        #     src_past_groups2 = torch.cat([src_all_groups2[1:], \
        #                                   src_max_groups2[:-1].repeat(1, (src.shape[0] - 1), 1, 1)], -1)
        #     src_all_groups2[1:] = self.cross_norm_3(self.cross_conv_3(src_past_groups2) + src_all_groups2[1:])
        #
        #     src_max_groups2 = torch.max(src_all_groups2, 1, keepdim=True).values
        #     src_past_groups2 = torch.cat([src_all_groups2[:-1], \
        #                                   src_max_groups2[1:].repeat(1, (src.shape[0] - 1), 1, 1)], -1)
        #     src_all_groups2[:-1] = self.cross_norm_4(self.cross_conv_4(src_past_groups2) + src_all_groups2[:-1])
        #
        #     src_inter_group_fusion2 = src_all_groups2.permute(1, 0, 2, 3).contiguous().flatten(1, 2)
        #
        #     src_inter_group_fusion = torch.concat([src_inter_group_fusion1,src_inter_group_fusion2],dim=1)
        #     src = torch.cat([src[:1],src_inter_group_fusion],0)
        
        return src, torch.cat(src[:1].chunk(self.num_groups,1),0)

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
        return self.forward_post(src,  pos)


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
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,dout=None,
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

    def forward(self, tgt,tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

def build_transformer3(args):
    return Transformer(
        config = args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        num_lidar_points = args.num_lidar_points,
        num_frames = args.num_frames,
        sequence_stride = args.get('sequence_stride',1),
        num_groups=args.num_groups,
    )






class VoxelSampler_denet(nn.Module):
    GAMMA = 1.1
    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5):
        super().__init__()
        
        self.voxel_size = voxel_size


        self.gen = PointToVoxel(
                        vsize_xyz=[voxel_size, voxel_size, pc_range[5]-pc_range[2]],
                        coors_range_xyz=pc_range,
                        num_point_features=num_point_features, 
                        max_num_voxels=50000,
                        max_num_points_per_voxel=max_points_per_voxel,
                        device=device
                    )

        self.gen1 = PointToVoxel(
            vsize_xyz=[voxel_size/2,voxel_size/2,voxel_size/2],
            coors_range_xyz=pc_range,
            num_point_features=num_point_features,
            max_num_voxels=50000,
            max_num_points_per_voxel=5,
            device=device
        )
        
        self.pc_start = torch.FloatTensor( pc_range[:2] ).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)
    def get_output_feature_dim(self):
        return self.num_point_features

    @staticmethod
    def cylindrical_pool(cur_points, cur_boxes, num_sample, gamma=1.):   
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample-len(cur_points)])
        cur_radiis = torch.norm(cur_boxes[:, 3:5]/2, dim=-1) * gamma
        dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample)
        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx).view(len(sampled_mask), num_sample, -1)

        sampled_points[sampled_mask==0, :] = 0

        return sampled_points


    def forward(self, batch_size, trajectory_rois,backward_rois, num_sample, batch_dict,valid_length):
        src1 = trajectory_rois.new_zeros(batch_size,trajectory_rois.shape[2],num_sample,5)
        src2 = list()

        rois = backward_rois.clone()
        rois[valid_length][:,:2] = (trajectory_rois[valid_length][:,:2]+backward_rois[valid_length][:,:2])/2

        for bs_idx in range(batch_size):
            
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
            cur_batch_boxes = rois[bs_idx]

            src2_points = list()
            for idx in range(trajectory_rois.shape[1]):
                gamma = self.GAMMA # ** (idx+1)

                time_mask = (cur_points[:,-1] - idx*0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask, :5].contiguous()

                cur_frame_boxes = cur_batch_boxes[idx]

                voxel, coords, num_points = self.gen(cur_time_points) 
                coords = coords[:, [2, 1]].contiguous()

                query_coords = ( cur_frame_boxes[:, :2] - self.pc_start ) // self.voxel_size

                radiis = torch.ceil(
                    torch.norm(cur_frame_boxes[:, 3:5] / 2, dim=-1) * gamma / self.voxel_size)

                dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :] )
                voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)
                num_points = num_points[voxel_mask]
                key_points = voxel[voxel_mask, :]
                point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)
                point_mask = num_points[:, None] > point_mask
                key_points = key_points[point_mask]
                if key_points.shape[0]!=0:
                    voxel_size = self.voxel_size/2
                    voxel,coords,num_points = self.gen1(key_points)
                    coords = coords[:, [2, 1]].contiguous()
                    query_coords = (cur_frame_boxes[:, :2] - self.pc_start) // voxel_size

                    dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :])

                    radiis = radiis*2
                    voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1)
                    # dist_mask = torch.all(dist < radiis[:, None, None], dim=-1)
                    sampled_mask, sampled_idx = torch.topk(voxel_mask.float(), min(num_sample,voxel_mask.shape[-1]))
                    sampled_voxel = torch.gather(voxel,0,sampled_idx.view(-1,1,1).repeat(1,voxel.shape[-2],voxel.shape[-1]))
                    sampled_voxel = sampled_voxel.view(sampled_mask.shape[0],-1,sampled_voxel.shape[-2],voxel.shape[-1]).permute(0,2,1,3)
                    sampled_voxel = sampled_voxel.reshape(sampled_voxel.shape[0],-1,sampled_voxel.shape[-1])
                    cur_radiis = torch.norm(cur_frame_boxes[:, 3:5]/2, dim=-1) * gamma
                    point_mask = ((sampled_voxel[:,:,:2]-cur_frame_boxes[:,None,:2])**2).sum(-1)<cur_radiis[:,None]**2
                    # point_mask = point_mask*(sampled_voxel[:,:,2]<cur_frame_boxes[:,None,2]+cur_frame_boxes[:,None,5]*0.6)
                    point_mask = point_mask*(sampled_voxel[:,:,2]<=(cur_frame_boxes[:,2]+cur_frame_boxes[:,5]*0.6)[:,None])
                    sampled_mask, sampled_idx = torch.topk(point_mask.float(), min(num_sample,point_mask.shape[-1]))
                    sampled_idx = sampled_idx[:,:,None].repeat(1, 1,sampled_voxel.shape[-1])
                    sampled_points = torch.gather(sampled_voxel, 1, sampled_idx).view(len(sampled_mask), num_sample, -1)
                    sampled_points[sampled_mask == 0, :] = 0
                else:
                    sampled_points = cur_points.new_zeros(cur_frame_boxes.shape[0],num_sample,cur_time_points.shape[-1])
                # voxel_mask = dist_mask.any(0)
                # dist_mask = dist_mask[:,voxel_mask]
                #
                # num_points = num_points[voxel_mask]
                # key_points = voxel[voxel_mask, :]
                #
                # point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)
                # key_points = dist_mask.new_zeros(dist_mask.shape[0],num_sample,cur_points.shape[-1])
                #
                #
                # point_mask = num_points[: , None] > point_mask
                # key_points = key_points[point_mask]
                # key_points = key_points[ torch.randperm(len(key_points)), :]
                #
                # key_points = self.cylindrical_pool(key_points, cur_frame_boxes, num_sample, gamma)
                src2_points.append(sampled_points)
                
            src2.append(torch.stack(src2_points))

        return torch.stack(src2).permute(0, 2, 1, 3, 4).flatten(2, 3)
        

def build_voxel_sampler_denet(device):
    return VoxelSampler_denet(
        device,
        voxel_size=0.4,
        pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
        max_points_per_voxel=32,
        num_point_features=5
    )


