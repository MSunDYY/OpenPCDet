import numpy as np
import spconv.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet import device
from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class SamplerVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size ,point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.grid_size = grid_size
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        voxel_mask = voxel_num_points>3
        key_points_all=[]
        for bc in range(batch_size):
            minimized_voxel = voxel_features[(coords[:,0]==bc)*voxel_mask][:,:2]
            key_points=minimized_voxel.reshape(-1,minimized_voxel.shape[-1])
            key_points=torch.cat([torch.full((key_points.shape[0],1),bc).to(device),key_points],dim=1)
            key_points_all.append(key_points)
        batch_dict['key_points_label'] = torch.cat(key_points_all,dim=0)[:,-1]
        batch_dict['key_points'] = torch.cat(key_points_all,dim=0)[:,:-1]
        batch_dict['points'] = batch_dict['points'][:,:-1]
        return batch_dict
