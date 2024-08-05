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
import math
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads,dropout = 0.0, ln=False):
        super(Attention, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.fc = nn.Linear(dim,dim*3)
        self.dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = nn.LayerNorm(dim)
            self.ln1 = nn.LayerNorm(dim)
        self.fc_o = nn.Linear(dim, dim)

    def forward(self, Q, drop=True):
        B = Q.shape[0]

        Q,K, V = self.fc(Q).chunk(3,-1)
        dim_split = self.dim_LIN // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        Q_ = Q_/math.sqrt(dim_split)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)), 2)
        A = self.dropout(A)
        if self.num_heads >= 2:
            temp = A.split(Q.size(0),dim=0)
            temp = torch.stack([tensor_ for tensor_ in temp], dim=0)
            weight = torch.mean(temp, dim=0)

        if drop:
            sampled_inds = torch.topk(weight.sum(1),A.shape[-1]//2,1)[1]
            sampled_inds_ = sampled_inds.repeat(self.num_heads,1)

            O = torch.gather(A, 1,sampled_inds_[:,:,None].repeat(1, 1,A.shape[-1])).bmm(V_)
            O = torch.concat(O.chunk(self.num_heads,0),-1)
            return self.fc_o(O), weight, sampled_inds
        else:
            O =torch.concat( A.bmm(V_).chunk(self.num_heads,0),dim=-1)
            return self.fc_o(O),weight,None

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads,dropout = 0.0, ln=False,batch_first = True):
        super(MultiheadAttention, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim,dim)
        self.fc_k = nn.Linear(dim,dim)
        self.fc_v = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_o = nn.Linear(dim, dim)

    def forward(self, Q,K,V, drop=True):
        B,T,D = Q.shape
        Q,K, V = self.fc_q(Q),self.fc_k(K) ,self.fc_v(V)
        dim_split = self.dim_LIN // self.num_heads
        Q = Q.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        K = K.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        V = V.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        Q = Q / math.sqrt(dim_split)
        A = torch.softmax(torch.bmm(Q,K.transpose(1,2)), 2)
        A = self.dropout(A)

        O = torch.bmm(A,V)
        O = O.view(B,self.num_heads,T,dim_split).transpose(1,2).contiguous().view(B,T,self.dim_LIN)
        O = self.fc_o(O)
        weight = A.view(B,self.num_heads,T,T).mean(dim=1)
        return O,weight


class SpatialMixerBlock(nn.Module):

    def __init__(self, channels,num_heads=8, dropout=0.0,batch_first=False):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, num_heads, dropout=dropout,batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
                               nn.Linear(channels, 2*channels),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(2*channels, channels),
                               )
    def forward(self, src,return_weight=True):
       
        src2,weight = self.mixer(src, src,src)
        
        src = src + self.dropout(src2)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer,weight
        else:
            return src_mixer


class SpatialDropBlock(nn.Module):

    def __init__(self, channels, config=None, dropout=0.0, batch_first=False):
        super().__init__()

        self.mixer = MultiheadAttention(channels,8,dropout,batch_first= True)
        # self.mixer = Attention(channels, 8,dropout=dropout )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, src, return_weight=False,drop=True):

        src2,weight = self.mixer(src,src,src)
        if drop:
            sampled_inds = torch.topk(weight.sum(1),weight.shape[-1]//2,1)[1]
            src =torch.gather(src,1,sampled_inds[:,:,None].repeat(1,1,src.shape[-1]))
            src2 = torch.gather(src2,1,sampled_inds[:,:,None].repeat(1,1,src2.shape[-1]))
        else:
            sampled_inds = None
        src = src+self.dropout(src2)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer, weight,sampled_inds
        else:
            return src_mixer


class CrossMixerBlock(nn.Module):

    def __init__(self, channels, config=None, dropout=0.0,batch_first = False):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout,batch_first=batch_first)

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

        # src = [torch.cat([torch.gather(token_list[i],1,batch_dict['roi_labels'].view(-1,1,1).repeat(1,1,self.d_model)-1,),src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        src = [torch.concat([token_list[i],src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        src = torch.cat(src,dim=0)

        # src = src.permute(1, 0, 2)
        memory,tokens = self.encoder(src,batch_dict,pos=pos)

        # memory = torch.cat(memory[0:1].chunk(self.num_groups,dim=1),0)
        return memory[:,:1], tokens,memory[:,1:]
    

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
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
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

        self.mlp_mixer_3d = SpatialDropBlock(
                        self.config.hidden_dim,
                        self.config.use_mlp_mixer,
                        dropout=dropout,
        )

        # self.point_attention = CrossMixerBlock(
        #     channels=96
        # )

        if self.layer_count<=self.config.enc_layers-1 and config.get('sampler',False) is not False:
            from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
            self.group = StackSAModuleMSG(radii=config.sampler.radius[self.layer_count-1],nsamples=config.sampler.nsample[self.layer_count-1],use_xyz=True,mlps=config.sampler.mlp[self.layer_count-1],use_spher=config.sampler.USE_SPHER)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,batch_dict,
                     pos: Optional[Tensor] = None):


        src_intra_group_fusion,weight,sampled_inds = self.mlp_mixer_3d(src[:,1:],return_weight=True,drop = self.layer_count>1)


        token = src[:,:1]

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
        src = torch.cat([token,src[:,1:]],1)

        src = torch.cat([src[:, :1], src_intra_group_fusion], 1)

        if self.layer_count > 1:

            # num_points = src.shape[0]-1
            # src_all_groups = src_intra_group_fusion.view((src_intra_group_fusion.shape[0])*self.num_groups,-1,src_intra_group_fusion.shape[-1])
            # # src_groups_list = src_all_groups.chunk(self.num_groups,0)
            # # src_groups_list = [src_all_groups[torch.arange(sampled_inds.shape[0])*self.num_groups+i] for i in range(self.num_groups)]
            #
            # src_all_groups = src_all_groups.unsqueeze(0)
            # src_max_groups = torch.max(src_all_groups, 1, keepdim=True).values
            # src_past_groups =  torch.cat([src_all_groups,\
            #      src_max_groups.repeat(1, src_intra_group_fusion.shape[0], 1, 1)], -1)
            # src_all_groups = self.cross_norm_1(self.cross_conv_1(src_past_groups) + src_all_groups)
            #
            #
            #
            # src_inter_group_fusion = src_all_groups.permute(1, 0, 2, 3).contiguous().flatten(1,2)
            #
            # weight = weight.sum(1)
            # # src = torch.cat([src[:1],src_intra_group_fusion],0)
            # sampled_inds = torch.topk(weight, dim=1, k=weight.shape[1] // 2)[1].transpose(0, 1)
            #
            # src_inter_group_fusion = torch.gather(src_inter_group_fusion, 0, sampled_inds[:, :, None].repeat(1, 1,
            #                                                                                                  src_inter_group_fusion.shape[
            #                                                                                                      -1]))
            
            batch_dict['src_idx'] = torch.gather(batch_dict['src_idx'],1,sampled_inds)

        return src, torch.cat(src[:,:1].chunk(self.num_groups,0),1)

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


class VoxelSampler(nn.Module):
    GAMMA = 1.05

    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5):
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

        self.pc_start = torch.FloatTensor(pc_range[:2]).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_size, trajectory_rois, num_sample, batch_dict):

        src = list()
        speed = torch.norm(trajectory_rois[..., -2:], dim=-1)
        for bs_idx in range(batch_size):

            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]
            cur_batch_boxes = trajectory_rois[bs_idx]
            src_points = list()

            for idx in range(trajectory_rois.shape[1]):
                gamma = torch.clamp((self.GAMMA *(1+speed[bs_idx,idx])) **(idx/5),max=2.5)# ** (idx+1)

                time_mask = (cur_points[:, -1] - idx * 0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask, :5].contiguous()

                cur_frame_boxes = cur_batch_boxes[idx]

                voxel, coords, num_points = self.gen(cur_time_points)
                coords = coords[:, [2, 1]].contiguous()

                query_coords = (cur_frame_boxes[:, :2] - self.pc_start) // self.voxel_size

                radiis = torch.ceil(
                    torch.norm(cur_frame_boxes[:, 3:5] / 2, dim=-1) * gamma  / self.voxel_size)

                dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :])

                voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)

                num_points = num_points[voxel_mask]
                key_points = voxel[voxel_mask, :]

                point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

                point_mask = num_points[:, None] > point_mask
                key_points = key_points[point_mask]
                key_points = key_points[torch.randperm(len(key_points)), :]

                key_points = self.cylindrical_pool(key_points, cur_frame_boxes, num_sample, gamma)

                src_points.append(key_points)

            src.append(torch.stack(src_points))

        return torch.stack(src).permute(0, 2, 1, 3, 4).flatten(2, 3)

    def cylindrical_pool(self, cur_points, cur_boxes, num_sample, gamma=1.):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])

        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample)

        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx).view(len(sampled_mask), num_sample, -1)

        sampled_points[sampled_mask == 0, :] = 0

        return sampled_points


class VoxelPointsSampler(nn.Module):
    GAMMA = 1.1

    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5,
                 config=None):
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
        self.num_recall_points=0
        self.num_gt_points=0
        self.num_recall_new = 0
        self.num_points = 0
        self.iteration = 0
        self.use_absolute_xyz = config.USE_ABSOLUTE_XYZ
        self.pc_start = torch.FloatTensor(pc_range[:2]).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.return_point_feature = config.ENABLE
        # self.point_emb = nn.Linear(sum([mlp[-1] for mlp in config.mlps])+(3 if self.use_absolute_xyz else 0),96)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_size, trajectory_rois, num_sample, batch_dict, start_idx=0, num_rois=None):
        src = list()
        query_points_list = list()
        src_idx_list = list()
        points_pre_list = list()
        idx_checkpoint = 0
        gamma = self.GAMMA

        cur_time_points = batch_dict['points'][batch_dict['points'][:, -1] == 0][:, :-1]

        for bs_idx in range(batch_size):
            cur_batch_points = cur_time_points[cur_time_points[:, 0] == bs_idx, 1:].contiguous()
            cur_batch_boxes = trajectory_rois[
                                  num_rois[bs_idx]:num_rois[bs_idx + 1]]


            voxel, coords, num_points = self.gen(cur_batch_points)
            num_voxel=num_points
            coords = coords[:, [2, 1]].contiguous()
            query_coords = (cur_batch_boxes[:, :2] - self.pc_start) / self.voxel_size

            radiis = torch.norm(cur_batch_boxes[:, 3:5] / 2, dim=-1)*gamma/ self.voxel_size

            dist = torch.norm(query_coords[:, None, :2] - coords[None, :, :],dim=-1)
            voxel_mask = (dist < radiis[:, None]).any(0)

            if not self.training:
                pre_roi = batch_dict['roi_list'][bs_idx,1:6]
                pre_roi[:,:,:2]-=pre_roi[:,:,7:]*torch.clamp(torch.arange(1,6,device=device),max=batch_dict['sample_idx'][0].item()+3)[:,None,None]
                pre_roi[:,:,3:5]*=1.0
                pre_roi = pre_roi.flatten(0,1)
                pre_roi = pre_roi[pre_roi[:,2]!=0]
                query_coords_pre = (pre_roi[:,:2] - self.pc_start) / self.voxel_size
                radiis_pre = torch.norm(pre_roi[:,3:5]/2,dim=-1)/self.voxel_size
                dist_pre = torch.norm(query_coords_pre[:,None,:2] - coords[None,:,:],dim=-1)
                voxel_mask_pre = (dist_pre<radiis_pre[:,None]).any(0)
                voxel_mask_pre = voxel_mask_pre*(~voxel_mask)
                key_points_pre = voxel[voxel_mask_pre,:]
                num_points_pre = num_points[voxel_mask_pre]
                point_mask_pre = torch.arange(self.k,device=device)[None,:].repeat(len(key_points_pre),1)
                point_mask_pre = point_mask_pre<num_points_pre[:,None]
                key_points_pre = key_points_pre[point_mask_pre]
                points_pre_list.append(key_points_pre)
            num_points = num_points[voxel_mask]
            key_points = voxel[voxel_mask, :]

            point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

            point_mask = num_points[:, None] > point_mask
            key_points = key_points[point_mask]
            key_points = key_points[torch.randperm(len(key_points)), :]

            key_points_raw = key_points


            key_points, src_idx ,query_points,points_pre = self.cylindrical_pool(key_points, cur_batch_boxes,
                                                        num_sample, gamma,
                                                        idx_checkpoint,pre_roi=None if self.training else pre_roi)
            idx_checkpoint+=query_points.shape[0]

            src.append(key_points)
            src_idx_list.append(src_idx)
            query_points_list.append(query_points)


            if not self.training:
                from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
                idx = torch.arange(voxel.shape[1],device=device)[None,:].repeat(voxel.shape[0],1)
                idx = idx<num_voxel[:,None]


                gt_points = roiaware_pool3d_utils.points_in_boxes_gpu(voxel[idx][None,:,:3],
                                                                      batch_dict['gt_boxes'][:, :, :7])
                self.num_gt_points += (gt_points >= 0).sum().item()

                recall_points = roiaware_pool3d_utils.points_in_boxes_gpu(key_points_raw[None,:,:3],
                                                                      batch_dict['gt_boxes'][:, :, :7])
                self.num_recall_points += (recall_points>=0).sum().item()
                recall_points_new = roiaware_pool3d_utils.points_in_boxes_gpu(torch.concat([key_points_pre,key_points_raw],dim=0)[None,:,:3],batch_dict['gt_boxes'][:,:,:7])
                self.num_recall_new +=(recall_points_new>=0).sum().item()
                self.num_points +=key_points_pre.shape[0]
                self.iteration+=1
            # src.append(torch.stack(src_points))
        return torch.concat(src, dim=0),torch.concat(src_idx_list,dim=0),torch.concat(query_points_list,dim=0),torch.concat(points_pre_list,dim=0) if not self.training else None

    def forward1(self, batch_size, trajectory_rois, num_sample, batch_dict, start_idx=0, num_rois=None):
        src = list()
        query_points_features_list = list()
        points_index_list = list()
        src_index_list = list()
        src_idx_checkpoint = 0
        points_idx_checkpoint = 0
        gamma = self.GAMMA
        num_frames = int((batch_dict['points'][:, -1].max() * 10).item()) + 1 if start_idx > 0 else 1
        for idx in range(num_frames - start_idx):
            cur_time_points = batch_dict['points'][batch_dict['points'][:, -1] == 0.1 * (idx + start_idx)][:, :-1]

            for bs_idx in range(batch_size):
                cur_batch_points = cur_time_points[cur_time_points[:, 0] == bs_idx, 1:].contiguous()
                cur_batch_boxes = trajectory_rois[
                                  num_rois[idx * batch_size + bs_idx]:num_rois[idx * batch_size + bs_idx + 1]]
                if start_idx == 1 or idx == 0:

                    voxel, coords, num_points = self.gen(cur_batch_points)
                    coords = coords[:, [2, 1]].contiguous()
                    query_coords = (cur_batch_boxes[:, :2] - self.pc_start) // self.voxel_size
                    radiis = torch.ceil(
                        torch.norm(cur_batch_boxes[:, 3:5] / 2, dim=-1) * gamma * 1.2 / self.voxel_size)

                    dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :])
                    voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)
                    num_points = num_points[voxel_mask]
                    key_points = voxel[voxel_mask, :]

                    point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

                    point_mask = num_points[:, None] > point_mask
                    key_points = key_points[point_mask]
                    key_points = key_points[torch.randperm(len(key_points)), :]
                else:
                    key_points = cur_batch_points
                if start_idx == 0 and idx == 0 and False:
                    root_data = '../../data/waymo/key_points/train/' if self.training else '../../data/waymo/key_points/train/'
                    os.makedirs(root_data + batch_dict['metadata'][0][:-4], exist_ok=True)
                    np.save(root_data + batch_dict['metadata'][0][:-4] + '/%04d.npy' % (
                        batch_dict['sample_idx'][0][-3:] if self.training else batch_dict['sample_idx'][0]),
                            key_points.cpu().numpy())

                key_points, src_index, points_index, query_points_features = self.cylindrical_pool_index(key_points,
                                                                                                         cur_batch_boxes,
                                                                                                         num_sample,
                                                                                                         gamma,
                                                                                                         )
                src_index += src_idx_checkpoint
                points_index[..., 0] += points_idx_checkpoint
                src_idx_checkpoint += points_index.shape[0]
                points_idx_checkpoint += src_index.shape[0]
                src.append(key_points)
                points_index_list.append(points_index)
                src_index_list.append(src_index)
                query_points_features_list.append(query_points_features)
            # src.append(torch.stack(src_points))
        query_points_bs_idx = torch.tensor([points.shape[0] for points in query_points_features_list],
                                           dtype=torch.int32, device=device)
        query_points_features = torch.concat(query_points_features_list, dim=0)
        points_index = torch.concat(points_index_list, dim=0)
        src_index = torch.concat(src_index_list, dim=0)
        for i in range(3):
            points_index[torch.prod(points_index[:, i + 1] == points_index[:, 0], dim=-1).bool(), i + 1] = -1
        return torch.concat(src, dim=0), src_index, points_index, query_points_features, query_points_bs_idx

    def cylindrical_pool_index(self, cur_points, cur_boxes, num_sample, gamma=1.):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])
        from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import ball_query
        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))
        # valid_points_mask = (point_mask.sum(0))!=0
        # cur_points,point_mask = cur_points[valid_points_mask],point_mask[:,valid_points_mask]

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample, 1)

        sampled, idx = torch.unique(sampled_idx, return_inverse=True)
        query_points = cur_points[sampled]
        query_points_features = self.set_abstraction(cur_points[None, :, :3].contiguous(),
                                                     cur_points[None, :, 3:].transpose(1, 2).contiguous(),
                                                     query_points[None, :, :3].contiguous())
        query_points_xyz = query_points_features[0][0]
        if self.use_absolute_xyz:
            query_points_features = torch.concat([query_points_xyz, query_points_features[1].transpose(1, 2)[0]],
                                                 dim=-1)
            query_points_features = self.point_emb(query_points_features)

        else:
            query_points_features = torch.concat([query_points_xyz, query_points_features[1].transpose(1, 2).squeeze()],
                                                 dim=-1)
        points_features = query_points_features[idx]

        sampled_mask = sampled_mask.bool()
        sampled_idx_ = (sampled_idx * sampled_mask).view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx_).view(len(sampled_mask), num_sample, -1)
        # unordered_points = sampled_points.flatten(0,1)
        # random_col = torch.randperm(unordered_points.shape[0],device=device)
        # unordered_points = unordered_points[random_col]
        # index_idx = ball_query(0.8,8,unordered_points[:,:3].contiguous(),torch.tensor([unordered_points.shape[0]],dtype=torch.int,device=device),sampled_points.flatten(0,1)[:,:3].contiguous(),torch.tensor([sampled_points.shape[0]*sampled_points.shape[1]],device=device,dtype=torch.int32))[0]
        # index_idx = random_col[index_idx.long()]
        # index_idx = index_idx.unflatten(0,(sampled_points.shape[0],-1))

        # index_idx = index_idx * sampled_mask[:,:,None]+idx_checkpoint
        sampled_points = sampled_points * sampled_mask[:, :, None]
        # idx = idx * sampled_mask + idx_checkpoint
        sampled_points = sampled_points
        src_idx = idx * sampled_mask
        points_idx = ball_query(0.0001, 4, src_idx.flatten(0, 1).float()[None, :, None].repeat(1, 1, 3),
                                torch.arange(query_points.shape[0], device=device).float()[None, :, None].repeat(1, 1,
                                                                                                                 3)).squeeze()

        points_idx = torch.stack([points_idx // src_idx.shape[1], points_idx % src_idx.shape[1]], -1)
        return sampled_points, src_idx, points_idx, query_points_features

    def cylindrical_pool(self, cur_points, cur_boxes, num_sample, gamma=1., idx_checkpoint=0,pre_roi=None):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])

        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)



        point_mask = (dis <= cur_radiis.unsqueeze(-1))


        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample, 1)




        sampled, idx = torch.unique(sampled_idx, return_inverse=True)
        query_points = cur_points[sampled]


        sampled_mask = sampled_mask.bool()
        sampled_idx_ = (sampled_idx * sampled_mask).view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx_).view(len(sampled_mask), num_sample, -1)
        idx = idx * sampled_mask + idx_checkpoint

        points_pre = cur_points[~point_mask.sum(0).bool()]

        return sampled_points, idx,query_points,points_pre
def build_voxel_sampler(device, return_point_feature=False):
    if not return_point_feature:
        return VoxelSampler(
            device,
            voxel_size=0.4,
            pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
            max_points_per_voxel=32,
            num_point_features=5,

        )
    else:
        return VoxelPointsSampler(
            device,
            voxel_size=0.4,
            pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
            max_points_per_voxel=32,
            num_point_features=5,
            config=return_point_feature,
        )



