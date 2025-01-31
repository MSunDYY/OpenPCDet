from os import getgrouplist
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from spconv.pytorch.utils import PointToVoxel
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_modules import PointnetSAModuleMSG
from pcdet import device
# from torch.utils.cpp_extension import load
# scatter = load(name='scatter', sources=['../cuda/scatter.cpp', '../cuda/scatter.cu'])
def unflatten(tensor,dim,sizes):
    shape = list(tensor.shape)
    shape = shape[:dim]+list(sizes)+shape[dim+1:]
    return tensor.reshape(shape)
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

class SpatialMixerBlockCompress(nn.Module):
    def __init__(self,hidden_dim,grid_size,dropout=0.0):
        super().__init__()
        self.grid_size = grid_size
        self.mixer_x = nn.Sequential(nn.Linear(hidden_dim*grid_size,hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.ReLU())
        self.norm_x = nn.LayerNorm(hidden_dim)

        self.mixer_y = nn.Sequential(nn.Linear(hidden_dim*grid_size,hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.ReLU())
        self.norm_y = nn.LayerNorm(hidden_dim)

        self.mixer_z = nn.Sequential(nn.Linear(hidden_dim*grid_size,hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.ReLU())
        self.norm_z = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim,2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim,hidden_dim)
        )
        self.norm_channel = nn.LayerNorm(hidden_dim)
        # self.mixer_z = MLP(input_dim=grid_size,hidden_dim=hidden_dim,output_dim=grid_size,num_layers=3)

    def forward(self,src):
        src3d = src.reshape(src.shape[0]*self.grid_size*self.grid_size,self.grid_size,src.shape[-1])
        src2d = self.norm_x(torch.max(src3d,dim=-2).values + self.mixer_x(src3d.flatten(1,2))).unflatten(0,(-1,self.grid_size))
        src1d = self.norm_y(torch.max(src2d,dim=-2).values + self.mixer_y(src2d.flatten(1,2))).unflatten(0,(-1,self.grid_size))
        src0d = self.norm_z(torch.max(src1d,dim=-2).values + self.mixer_z(src1d.flatten(1,2)))

        src0d = self.norm_channel(src0d+self.ffn(src0d))
        return src0d
class SpatialMixerBlock(nn.Module):

    def __init__(self, channels,config=None, dropout=0.0):
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
    def forward(self, src,return_weight=False):
       
        src2,weight = self.mixer(src, src, src)
        
        src = src + self.dropout(src2)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer,weight
        else:
            return src_mixer


class CrossMixerBlock(nn.Module):

    def __init__(self, channels, config=None, dropout=0.0):
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
    def emb_pos(self,tensor,pos):
        return tensor if pos is None else tensor+pos
    def forward(self, query,key,pos_q=None,pos_k=None,return_weight=False):
        
        query = query if pos_q is None else query+pos_q
       
        src2, weight = self.mixer(self.emb_pos(query,pos_q), self.emb_pos(key,pos_k), key)

        query = query + self.dropout(src2)
        src_mixer = self.norm(query)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer, weight
        else:
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
        self.token = nn.Parameter(torch.zeros(self.num_groups, 1, d_model))

        
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

    def forward(self, src,batch_dict, pos=None):

        BS, N, C = src.shape
        if not pos is None:
            pos = pos.permute(1, 0, 2)
            

        token_list = [self.token[i:(i+1)].repeat(BS,1,1) for i in range(self.num_groups)]

        src = [torch.cat([token_list[i],src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        # src = [src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points] for i in range(self.num_groups)]
        src = torch.cat(src,dim=0)

        src = src.permute(1, 0, 2)
        memory,tokens = self.encoder(src,batch_dict,pos=pos)

        memory = torch.cat(memory[0:1].chunk(self.num_groups,dim=1),0)
        return memory, tokens
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,batch_dict,
                pos: Optional[Tensor] = None):

        token_list = []
        output = src
        for layer in self.layers:
            output,tokens = layer(output,batch_dict,pos=pos)
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
        self.point_feature = 96
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = vector_attention(d_model, nhead=4)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.layer_count <= self.config.enc_layers-1:
            self.cross_conv_1 = nn.Linear(d_model * 2, d_model)
            self.cross_norm_1 = nn.LayerNorm(d_model)
            self.cross_conv_2 = nn.Linear(d_model * 2, d_model)
            self.cross_norm_2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(
                        self.config.hidden_dim, 
                        self.config.use_mlp_mixer
        )

        self.point_attention = CrossMixerBlock(
            channels=96
        )

        if self.layer_count<=self.config.enc_layers-1 and config.get('sampler',False) is not False:
            from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
            self.group = StackSAModuleMSG(radii=config.sampler.radius[self.layer_count-1],nsamples=config.sampler.nsample[self.layer_count-1],use_xyz=True,mlps=config.sampler.mlp[self.layer_count-1],use_spher=config.sampler.USE_SPHER)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,batch_dict,
                     pos: Optional[Tensor] = None):
        query_points_features = batch_dict['query_points_features'+str(self.layer_count)]
        src_idx = batch_dict['src_idx'+str(self.layer_count)]
        query_points_idx_bs = batch_dict['query_points_bs_idx'+str(self.layer_count)]
        src_intra_group_fusion,weight = self.mlp_mixer_3d(src[1:],return_weight=True)
        num_rois = batch_dict['num_rois']
        # src_intra_group_fusion = torch.concat([src_intra_group_fusion, src[1:,:,self.config.hidden_dim:]], dim=-1)

        token = src[:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion

      
        src_summary = self.self_attn(token, key,src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = torch.cat([token,src[1:]],0)


        if self.layer_count <= self.config.enc_layers-1:
            weight = weight.sum(1)
            # src = torch.cat([src[:1],src_intra_group_fusion],0)
            sampled_inds = torch.topk(weight, dim=1, k=weight.shape[1] // 2)[1].transpose(0,1)
            new_src_idx = torch.gather(src_idx, 0, sampled_inds)
            new_query_idx = [torch.unique(new_src_idx[:,num_rois[i]:num_rois[i+1]]) for i in range(num_rois.shape[0]-1)]

            # new_src_idx = new_src_idx.transpose(0, 1).reshape(batch_dict['num_frames'] * batch_dict['batch_size'], -1)

            # new_query_idx = [torch.unique(new_src_idx[i]) for i in range(new_src_idx.shape[0])]
            new_query_idx_bs = torch.tensor([i.shape[0] for i in new_query_idx],device=device,dtype=torch.int32)
            new_query_idx = torch.concat(new_query_idx,dim=0)
            # num_new_query = max([n.shape[0] for n in new_query_idx])
            # new_query_idx = torch.stack([F.pad(n, (0, num_new_query - n.shape[0])) for n in new_query_idx])
            new_query_xyz, new_query_features = torch.split(torch.gather(query_points_features, 0,
                                                                     new_query_idx[:,  None].repeat(1,query_points_features.shape[-1])),
                                                        [3, query_points_features.shape[-1] - 3], dim=-1)
            new_query_features = self.group(query_points_features[..., :3].contiguous(),query_points_idx_bs,
                                                   new_query_xyz.contiguous(),new_query_idx_bs,query_points_features[..., 3:].contiguous())
            # new_query_features = self.point_attention(new_query_features.reshape(1, -1, new_query_features.shape[-1]),
            #                                         new_query_features_grouped.permute(3, 0, 2, 1).flatten(1, 2)).reshape(
            #     -1, new_query_xyz.shape[1], new_query_features.shape[-1])
            new_query_points_features = torch.concat(new_query_features,dim=-1)
            # new_query_points_features = torch.zeros_like(query_points_features)
            new_src_idx = (torch.searchsorted(new_query_idx,new_src_idx,right=True)-1)
            # print('%02d  %02d  %02d'%(new_src_idx.min().item(),new_src_idx.max().item(),new_query_points_features.shape[0]))
            assert new_src_idx.min().item()==0,new_src_idx.max().item()==new_query_points_features.shape[0]-1

            # new_query_points_features.scatter_(0, new_query_idx[:,  None].repeat(1,
            #                                                                      new_query_points_features.shape[-1]),
            #                                    torch.concat([new_query_xyz,new_query_features], dim=-1))
            # new_query_points_features = torch.concat([new_query_xyz,new_query_features],dim=-1)
            batch_dict['query_points_features' + str(self.layer_count + 1)] = new_query_points_features
            batch_dict['src_idx'+str(self.layer_count+1)] = new_src_idx
            batch_dict['query_points_bs_idx'+str(self.layer_count+1)] = new_query_idx_bs
            # src_point_feature = torch.gather(new_query_points_features,0,new_src_idx.reshape(-1,1).repeat(1,new_query_points_features.shape[-1]))
            # src_point_feature = src_point_feature.reshape(-1,sampled_inds.shape[0],src_point_feature.shape[-1]).transpose(0,1)

            src_intra_group_fusion = torch.gather(src_intra_group_fusion, 0, sampled_inds[:, :, None].repeat(1, 1,
                                                                                                             src_intra_group_fusion.shape[-1]))
            num_points = src.shape[0]-1
            src_all_groups = src_intra_group_fusion.view((src_intra_group_fusion.shape[0])*self.num_groups,-1,src_intra_group_fusion.shape[-1])
            # src_groups_list = src_all_groups.chunk(self.num_groups,0)
            src_groups_list = [src_all_groups[torch.arange(sampled_inds.shape[0])*self.num_groups+i] for i in range(self.num_groups)]

            src_all_groups = torch.stack(src_groups_list, 0)
            src_all_groups = src_all_groups[...,:self.config.hidden_dim]
            src_max_groups = torch.max(src_all_groups, 1, keepdim=True).values
            src_past_groups =  torch.cat([src_all_groups,\
                 src_max_groups.repeat(1, src_intra_group_fusion.shape[0], 1, 1)], -1)
            src_all_groups = self.cross_norm_1(self.cross_conv_1(src_past_groups) + src_all_groups)

            # src_max_groups = torch.max(src_all_groups, 1, keepdim=True).values
            # src_past_groups =  torch.cat([src_all_groups[:-1],\
            #      src_max_groups[1:].repeat(1, src_intra_group_fusion.shape[0], 1, 1)], -1)
            # src_all_groups[:-1] = self.cross_norm_2(self.cross_conv_2(src_past_groups) + src_all_groups[:-1])

            src_inter_group_fusion = src_all_groups.permute(1, 0, 2, 3).contiguous().flatten(1,2)
            
            src = torch.cat([src[:1],src_inter_group_fusion],0)

        else:
            new_src_idx = src_idx.flatten(0,1)

            # new_query_idx = [torch.unique(new_query_idx[i]) for i in range(new_query_idx.shape[0])]
            # num_new_query = max([n.shape[0] for n in new_query_idx])
            # new_query_idx = torch.stack([F.pad(n, (0, num_new_query - n.shape[0])) for n in new_query_idx])
            new_src_xyz, new_src_features = torch.split(torch.gather(query_points_features, 0,
                                                                         new_src_idx[:, None].repeat(1, query_points_features.shape[-1])),
                                                            [3, query_points_features.shape[-1] - 3], dim=-1)
            new_src_points_features = new_src_features.reshape(src.shape[0]-1,-1,new_src_features.shape[-1])
            new_src_xyz = new_src_xyz.reshape(src.shape[0]-1,-1,new_src_xyz.shape[-1])
            # new_src_points_features = unflatten(new_src_points_features,dim=0,sizes=(batch_dict['num_frames'],-1))[0]
            # new_src_xyz = unflatten(new_src_xyz,dim=0,sizes = (batch_dict['num_frames'],-1))[0]
            batch_dict['final_src_points_features'] = new_src_points_features
            batch_dict['final_src_xyz'] = new_src_xyz
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

    def forward(self, src,batch_dict,
                pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src,batch_dict,  pos)


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

def build_transformer(args):
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


