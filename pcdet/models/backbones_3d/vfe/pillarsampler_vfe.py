import numpy as np
import spconv.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet import device
from .vfe_template import VFETemplate
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
from tools.visual_utils.open3d_vis_utils import draw_scenes


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


class PillarSamplerVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.use_norm = model_cfg.USE_NORM
        self.use_absoluto_xyz = self.model_cfg.get('USE_ABSOLUTE_XYZ', False)
        self.num_point_features += 6 if self.use_absoluto_xyz else 3
        self.num_features_layers = [self.num_point_features] + list(model_cfg.NUM_FILTERS)

        pfn = []
        for i in range(len(self.num_features_layers) - 1):
            in_channels = self.num_features_layers[i]
            out_channels = self.num_features_layers[i + 1]
            pfn.append(PFNLayer(in_channels, out_channels, use_norm=self.use_norm,
                                last_layer=True if i == len(self.num_features_layers) - 2 else False))
            self.pfn = nn.ModuleList(pfn)
        cur_num_features = self.num_features_layers[-1]
        out_num_features = [int(cur_num_features / 2), int(cur_num_features / 4), int(cur_num_features / 4)]
        self.conv = nn.ModuleList([
            nn.ModuleList([nn.Sequential(
                nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
                nn.Conv2d(in_channels=cur_num_features * 2, out_channels=out_num_features[0],
                          kernel_size=2 * i + 3, stride=1),
                nn.BatchNorm2d(out_num_features[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ) for i in range(3)]),
            nn.ModuleList([nn.Sequential(
                nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
                nn.Conv2d(in_channels=out_num_features[0] * 3 * 2, out_channels=out_num_features[1],
                          kernel_size=2 * i + 3, stride=1),
                nn.BatchNorm2d(out_num_features[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ) for i in range(3)]),
            nn.ModuleList([nn.Sequential(
                nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
                nn.Conv2d(in_channels=out_num_features[1] * 3 * 2, out_channels=out_num_features[2],
                          kernel_size=2 * i + 3, stride=1),
                nn.BatchNorm2d(out_num_features[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ) for i in range(3)])
        ])
        self.classifier = nn.Linear(in_features=out_num_features[-1] * 3, out_features=2)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.offset_x = self.voxel_x / 2 + point_cloud_range[0]
        self.offset_y = self.voxel_y / 2 + point_cloud_range[1]
        self.offset_z = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_point_features

    def get_label(self, batch_dict):
        voxels = batch_dict['voxels']

    def forward(self, batch_dict, **kwargs):
        torch.cuda.empty_cache()
        voxels, voxel_num_points, coordinates = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxels[:, :, :3].sum(dim=-2) / (voxel_num_points.reshape(-1, 1).repeat(1, 3))
        f_cluster = voxels[:, :, :3] - points_mean.unsqueeze(-2)
        f_center = torch.zeros_like(voxels[:, :, :3])
        f_center[:, :, 0] = voxels[:, :, 0] - (coordinates[:, 4].unsqueeze(1) * self.voxel_x + self.offset_x)
        f_center[:, :, 1] = voxels[:, :, 1] - (coordinates[:, 3].unsqueeze(1) * self.voxel_y + self.offset_y)
        f_center[:, :, 2] = voxels[:, :, 2] - (coordinates[:, 2].unsqueeze(1) * self.voxel_z + self.offset_z)
        if self.use_absoluto_xyz:
            features = [voxels[:, :, :5], f_cluster, f_center]


        features = torch.cat(features, dim=-1)
        padding = torch.arange(voxels.shape[-2]).reshape((1, -1)).repeat(voxels.shape[0], 1).to(device)
        mask = padding < voxel_num_points.reshape((-1, 1))
        features *= mask.unsqueeze(-1)
        for pfn in self.pfn:
            features = pfn(features)

        B = int(coordinates[:, 0].max().item()) + 1
        voxels[:, :, -2] *= 10
        F = int(coordinates[:, 1].max().item()) + 1
        H, W = self.grid_size[:2]

        pillar_map = torch.zeros(F, B, H, W, features.shape[-1]).to(device)
        coordinates = [coordinates[:, 1].long(), coordinates[:, 0].long(), coordinates[:, -1].long(),
                       coordinates[:, -2].long()]

        pillar_map[coordinates] = features.squeeze()
        pillar_map = torch.permute(pillar_map, (0, 1, 4, 2, 3))
        for convs in self.conv:
            cur_pillar = pillar_map[:-1]
            pre_pillar = pillar_map[1:]
            concated_feature = torch.concat(
                [cur_pillar, pre_pillar], dim=-3)
            features = []
            for conv in convs:
                features.append(conv(concated_feature.reshape((-1, concated_feature.shape[-3], H, W))))
            pillar_map = torch.concat(features, dim=-3)
            pillar_map = pillar_map.reshape((-1, B, pillar_map.shape[-3], H, W))

        pillar_map = torch.permute(pillar_map, (0, 1, 3, 4, 2))
        pred_label = self.classifier(pillar_map)
        pred_label = torch.permute(pred_label.reshape((B, H, W, -1)), (3, 0, 1, 2))
        interest_num = (coordinates[0] < 2).sum()
        pred_label = pred_label[[coor[:interest_num] for coor in coordinates]]

        key_pillar_label = torch.zeros((F, B, H, W)).to(device)
        key_pillar_label[coordinates] = (voxels[:, :, 6].sum(dim=-1) > 0).type(torch.float)
        key_pillar_label = key_pillar_label[[coor[:interest_num] for coor in coordinates]]

        batch_dict['key_pillars_pred'] = pred_label
        batch_dict['key_pillars_label'] = key_pillar_label
        return batch_dict
