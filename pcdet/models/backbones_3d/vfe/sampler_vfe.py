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

        torch.cuda.empty_cache()
        voxels, voxel_num_points, coordinates = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        batch_label = torch.zeros(voxels.shape[0],voxels.shape[1],1).to(device)
        batch_label[:] = coordinates[:,0:1].unsqueeze(1).expand(-1,voxels.shape[-2],-1)
        voxels = torch.cat((batch_label,voxels),dim=-1)

        L,W,H=self.grid_size
        B=batch_dict['batch_size']
        dense_voxel = torch.zeros((B,L, W, H, voxels.shape[-1] * voxels.shape[-2])).cuda()
        coordinates_np = coordinates.to('cpu').numpy()

        split = [coordinates_np[:,0],coordinates_np[:,-1],coordinates_np[:,-2],coordinates_np[:,-3]]
        dense_voxel[split] = voxels.reshape(voxels.shape[0],-1)
        B, H, W, D, C = dense_voxel.size()
        dense_voxel = dense_voxel.reshape((B*H*W,D*voxels.shape[-2],-1))
        points_num_pillar = ((dense_voxel[:, :, 1] != 0) + (dense_voxel[:, :, 2] != 0) + (
                dense_voxel[:, :, 3] != 0)).sum(axis=-1)
        dense_voxel = dense_voxel[points_num_pillar!=0]

        point_features = [dense_voxel[:, :, :-1]]
        num_point_features = dense_voxel.shape[-1]
        mean_z = torch.sum(dense_voxel[:, :, 3], dim=-1) / points_num_pillar[points_num_pillar!=0]
        mean_z = mean_z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        point_features.append(mean_z.expand(-1, dense_voxel.shape[1], -1))
        num_point_features += 1

        height = torch.max(dense_voxel[:, :, 3], dim=-1)[0] - \
                 torch.min(dense_voxel[:, :, 3], dim=-1)[0]
        height = height.unsqueeze(dim=-1).unsqueeze(dim=-1)
        point_features.append(height.expand(-1, dense_voxel.shape[1], -1))
        num_point_features += 1

        point_features.append(dense_voxel[:, :, -1].unsqueeze(-1))
        point_features = torch.cat(point_features, axis=-1)


        point_features = point_features.reshape(-1,D,voxels.shape[-2] ,point_features.shape[-1])

        point_mask = ((point_features[:,:,:,1]!=0)+(point_features[:,:,:,2]!=0)+(point_features[:,:,:,3]!=0))
        point_num_voxel = point_mask.sum(axis=-1)

        voxel_features = point_features[point_num_voxel>0]

        voxel_mask = voxel_num_points>3

        point_features = point_features[point_mask]

        minimized_voxel = voxel_features[voxel_mask][:,:2]
        key_points=minimized_voxel.reshape(-1,minimized_voxel.shape[-1])

        batch_dict['key_points_label'] = key_points[:,-1]
        batch_dict['key_points'] = key_points[:,:-1]
        batch_dict['points'] = point_features[:,:-1]

        return batch_dict
