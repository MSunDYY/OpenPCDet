import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from pcdet import device
from pcdet.models.backbones_3d.spconv_backbone import SparseBasicBlock
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
from tools.visual_utils.open3d_vis_utils import draw_scenes
from pcdet.ops.deformDETR.modules.ms_deform_attn import MSDeformAttn
from pcdet.ops.deformDETR.functions import MSDeformAttnFunction
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import ball_query, grouping_operation
from spconv.pytorch.functional import sparse_add
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.utils import PointToVoxel
from pcdet.models.backbones_3d.spconv_backbone import post_act_block, SparseBasicBlock
from pcdet.utils.spconv_utils import replace_feature
from ...models.model_utils.mppnet_utils import MLP


class DeformableTransformerCrossAttention(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.2, n_heads=1, n_points=2, n_levels=1, out_sample_loc=False):
        super().__init__()
        self.n_points = n_points
        self.cross_atten = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.drop_out = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.drop_out(self.activation(self.linear1(src))))
        src = src + self.drop_out(src)
        src = self.norm2(src)
        return src

    def forward(self, tgt, src, query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        query = self.cross_atten(tgt, reference_points, src, spatial_shapes, level_start_index)
        query = tgt + self.drop_out(query)
        query = self.norm1(query)
        src = self.forward_ffn(src)


class AttentionAggregation(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, n_points=(4, 2, 1), n_heads=1, n_levels=1):
        super().__init__()

        self.im2col_step = 64
        self.n_points = n_points
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.speed_est1 = nn.Linear(in_features=in_channels, out_features=2 * n_points[0] * n_levels * n_heads)
        self.speed_weight1 = nn.Linear(in_features=in_channels, out_features=n_points[0] * n_levels * n_heads)
        self.speed_est2 = nn.Linear(in_features=out_channels, out_features=2 * n_points[1] * n_levels * n_heads)
        self.speed_weight2 = nn.Linear(in_features=out_channels, out_features=n_points[1] * n_levels * n_heads)
        self.speed_est3 = nn.Linear(in_features=out_channels, out_features=2 * n_points[2] * n_levels * n_heads)
        self.speed_weight3 = nn.Linear(in_features=out_channels, out_features=n_points[2] * n_levels * n_heads)
        self.deform_attention = nn.ModuleList([])
        for n_point in n_points:
            self.deform_attention.append(
                DeformableTransformerCrossAttention(in_channels, out_channels, 0.2, n_heads, n_point, n_levels))

    def forward(self, cur_features, pre_features, cur_indices, pre_indices, spatial_shape):
        B = cur_indices[:, 0].max().item() + 1
        F = cur_indices[:, 1].max().item() + 1
        num_cur = [[((cur_indices[:, 0] == b) & (cur_indices[:, 1] == f)).sum() for f in range(F)] for b in range(B)]
        num_pre = [[((pre_indices[:, 0] == b) & (pre_indices[:, 1] == f + 1)).sum() for f in range(F)] for b in
                   range(B)]
        max_num_cur = max(max(num) for num in num_cur)
        max_num_pre = max(max(num) for num in num_pre)
        cur_features_dense = cur_features.new_zeros((B * F, max_num_cur, cur_features.shape[-1]))
        pre_features_dense = cur_features.new_zeros((B * F, spatial_shape[-2], spatial_shape[-1], pre_features.shape[-1]))
        cur_indices_dense = cur_features.new_zeros((B * F, max_num_cur, 2))
        pre_indices_dense = cur_features.new_zeros((B * F, max_num_pre, 2))

        for b in range(B):
            for f in range(F):
                cur_mask = (cur_indices[:, 0] == b) & (cur_indices[:, 1] == f)
                pre_mask = (pre_indices[:, 0] == b) & (pre_indices[:, 1] == f + 1)
                cur_num = cur_mask.sum().item()
                pre_num = pre_mask.sum().item()
                cur_features_dense[b * F + f][:cur_num] = cur_features[cur_mask]
                cur_indices_dense[b * F + f][:cur_num] = cur_indices[cur_mask][:, -2:]
                pre_indices_dense[b * F + f][:pre_num] = pre_indices[pre_mask][:, -2:]
                pre_features_dense[b * F + f][
                    pre_indices[pre_mask][:, -2].long(), pre_indices[pre_mask][:, -1].long()] = pre_features[pre_mask]
        pre_features_dense = pre_features_dense.reshape((B * F, -1, pre_features_dense.shape[-1]))

        speed_est = pre_features_dense.new_features((B * F, max_num_cur, 2))
        pre_features_dense = pre_features_dense.reshape(B * F, pre_features_dense.shape[1], self.n_heads, -1)
        spatial_shape = spatial_shape[None, :]
        input_level_start_index = torch.tensor([0]).to(device)
        for i, n_point in enumerate(self.n_points):
            speed_delta = getattr(self, 'speed_est%d' % (i + 1))(cur_features_dense).reshape(
                (B * F, max_num_cur, self.n_heads, self.n_levels, n_point, 2))

            speed_weight = getattr(self, 'speed_weight%d' % (i + 1))(cur_features_dense).reshape(B * F, max_num_cur,
                                                                                                 self.n_heads,
                                                                                                 self.n_levels *
                                                                                                 n_point)
            speed_weight = Functional.softmax(speed_weight, -1).reshape(B * F, max_num_cur, self.n_heads, self.n_levels,
                                                                        n_point)
            ref_points = cur_indices_dense[:, :, None, :] / spatial_shape[None, None, :]
            sampling_offset = speed_est[:, :, None, None, None, :] + speed_delta

            max_speed = speed_weight.max(dim=-1, keepdim=True)[1].unsqueeze(-1)
            max_speed = torch.concat([max_speed, max_speed], -1)
            speed_est = speed_est + torch.gather(speed_delta, 4, max_speed).squeeze()
            sampling_locations = ref_points[:, :, None, :, None, :] + sampling_offset * 0.1 / spatial_shape[None, None,
                                                                                              None,
                                                                                              : None, :]

            cur_features_dense = MSDeformAttnFunction.apply(pre_features_dense, spatial_shape, input_level_start_index,
                                                            sampling_locations,
                                                            speed_weight,
                                                            self.im2col_step)

        for b in range(B):
            for f in range(F):
                cur_mask = (cur_indices[:, 0] == b) & (cur_indices[:, 1] == f)
                cur_features[cur_mask] = cur_features_dense[b * F + f][:cur_mask.sum()]
        speed_est = torch.concat([speed_est[b * F + f][:num_cur[b][f]] for f in range(F) for b in range(B)])
        return cur_features, speed_est


class SEDBlock(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_SBB, norm_fn, indice_key):
        super(SEDBlock, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        # block_list = [first_block if stride > 1 else nn.Identity()]
        # for _ in range(num_SBB):
        #     block_list.append(
        #         SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))
        # block_list.append(spconv.SparseMaxPool3d(kernel_size=(3,)))
        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)


class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, norm_fn, indice_key):
        super().__init__()

        # assert down_stride[0] == 1  # hard code
        assert len(down_kernel_size) == len(down_stride) == len(num_SBB)

        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.pool_inv = nn.ModuleList()
        self.max_pool = spconv.SparseMaxPool3d(kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        num_features = [dim]
        for i in range(len(down_stride)):
            num_features.append(num_features[i] * down_stride[i])
        for idx in range(len(down_stride)):
            self.encoder.append(
                spconv.SparseSequential(
                    post_act_block(
                        num_features[idx], num_features[idx + 1], kernel_size=(3, 3, 3),
                        stride=(1, down_stride[idx], down_stride[idx]), padding=(1, 1, 1),
                        norm_fn=norm_fn, indice_key=f'spconv_{idx}', conv_type='spconv'),
                    post_act_block(num_features[idx + 1], num_features[idx + 1], kernel_size=(3, 3, 3), stride=1,
                                   padding=(1, 1, 1), norm_fn=norm_fn, indice_key=f'subm_spa{idx}',
                                   conv_type='subm'), ))
            # post_act_block(num_features[idx], num_features[idx], kernel_size=(3, 1, 1), stride=1,
            #                padding=(1, 0, 0), norm_fn=norm_fn, indice_key=f'subm_tem{idx}',
            #                conv_type='subm'),

            # self.max_pool.append(spconv.SparseMaxPool3d(kernel_size=(5,1,1),stride=(2,1,1),padding=(1,0,0),indice_key=f'max_pool_{idx}'))
        downsample_times = len(down_stride  )
        
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.conv2d = nn.ModuleList()
        self.tranconv = nn.ModuleList()

        for idx, kernel_size in enumerate(down_kernel_size):
            self.decoder.append(
                spconv.SparseSequential(

                    post_act_block(num_features[-idx - 1], num_features[-idx - 1], kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(0, 1, 1), norm_fn=norm_fn, indice_key=f'tem_down_{idx}',
                                   conv_type='spconv'),
                    post_act_block(num_features[-idx - 1], num_features[-idx - 1], kernel_size=(3, 3, 3),
                                   norm_fn=norm_fn, stride=(2, 1, 1), padding=1, conv_type='spconv',
                                   indice_key=f'tem_flatten_{idx}')
                )

            )
            self.tranconv.append(spconv.SparseSequential(
                spconv.SparseConvTranspose3d(num_features[-idx - 1], num_features[-idx - 2], kernel_size=(1, 3, 3),
                                             stride=(1, 2, 2), padding=(0, 1, 1)),
                norm_fn(num_features[-idx - 2]),
                nn.ReLU()
            ))
            self.conv2d.append(spconv.SparseSequential(
                spconv.SubMConv3d(in_channels=num_features[-idx - 2] * 2, out_channels=num_features[-idx - 2], stride=1,
                                  kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                norm_fn(num_features[-idx - 2]),
                nn.ReLU()
            ))
            self.decoder_norm.append(norm_fn(num_features[-idx - 2]))
            # self.pool.append(
            #     spconv.SparseMaxPool3d(kernel_size=3, stride=1, padding=1, indice_key=f'{indice_key}_pool_{downsample_times-idx}'))
            # self.pool_inv.append(
            #     spconv.SparseInverseConv3d(dim, dim, 3, indice_key = f'{indice_key}_pool_{downsample_times-idx}',bias=False)
            # )
        self.decoder.append(
            spconv.SparseSequential(

                post_act_block(num_features[0], num_features[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), norm_fn=norm_fn, indice_key=f'tem_down_{idx + 1}',
                               conv_type='spconv'),
                post_act_block(num_features[0], num_features[0], kernel_size=(3, 3, 3), norm_fn=norm_fn,
                               stride=(2, 1, 1), padding=1, conv_type='spconv', indice_key=f'tem_flatten_{idx + 1}')
            )
        )

    def forward(self, x):
        features = [x]

        for idx, conv in enumerate(self.encoder):
            x = conv(x)

            features.append(x)

        #

        x = features[-1]
        x = self.decoder[0](x)
        for idx in range(len(self.decoder) - 1):
            # for deconv, conv2d, norm, up_x in zip(self.decoder, self.conv2d, self.decoder_norm, features[:-1][::-1]):

            up_x = features[-idx - 2]
            up_x = self.decoder[idx + 1](up_x)
            index = up_x.indices.long()

            x = self.tranconv[idx](x)
            x = replace_feature(up_x,
                                torch.concat([x.dense().permute(0, 2, 3, 4, 1)[
                                                           index[:, 0], index[:, 1], index[:, 2], index[:,
                                                                                                  3]] , up_x.features],dim=-1))
            x=self.conv2d[idx](x)
            # x = self.conv2d[idx](x)

            # up_x = replace_feature(x, norm(x.features))
            # idx+=1
        return x


class TemperalUpConv(spconv.SparseModule):
    def __init__(self, inplaces, planes, stride=(2, 2), bias=None, norm_fn=None, downsample=None, indice_key=None):
        super().__init__()
        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SparseSequential(
            spconv.SparseConvTranspose3d(inplaces, planes, 3, stride=(1, 1, 1), padding=1),
            norm_fn(planes),
            nn.ReLU())
        self.conv2 = spconv.SparseSequential(
            spconv.SparseConvTranspose3d(inplaces, planes, kernel_size=(3, 3, 3), stride=(1, stride[0], stride[0]),
                                         padding=(1, 1, 1), bias=False,
                                         ),
            norm_fn(planes),
            nn.ReLU()
        )
        self.conv3 = spconv.SparseSequential(
            spconv.SparseConvTranspose3d(inplaces, planes, kernel_size=(3, 3, 3), stride=(1, stride[1], stride[1]),
                                         padding=1, bias=False,
                                         ),
            norm_fn(planes),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, stride[0], stride[0]),
                                         padding=(1, 1, 1), bias=False),
            norm_fn(planes),
            nn.ReLU()
        )

    def forward(self, x):
        deconv1 = self.conv1(x[0])
        deconv2 = self.conv2(x[1])
        deconv3 = self.conv3(x[2])
        features = sparse_concat(deconv1, deconv2, deconv3, dim=-1)

        return features


class TemperalDownConv(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=(2, 2), bias=None, norm_fn=None, downsample=None, indice_key=None):
        super().__init__()
        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None

        self.conv1 = spconv.SparseSequential(
            post_act_block(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), norm_fn=norm_fn,
                           padding=(0, 1, 1), indice_key='spconv1',
                           conv_type='spconv'),
            post_act_block(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), norm_fn=norm_fn,
                           padding=(0, 1, 1),
                           indice_key='spconv2',
                           conv_type='spconv'),

            post_act_block(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), norm_fn=norm_fn,
                           padding=(0, 1, 1),
                           indice_key='spconv2',
                           conv_type='inverseconv'),
            post_act_block(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 2, 2), norm_fn=norm_fn,
                           padding=(0, 1, 1),
                           indice_key='spconv1',
                           conv_type='inverseconv'),
            # SparseBasicBlock(planes,planes,norm_fn=norm_fn,indice_key='stem'),
            # SparseBasicBlock(planes, planes, norm_fn=norm_fn, indice_key='stem'),
        )

        self.max_pool = spconv.SparseMaxPool3d(kernel_size=(5, 1, 1), stride=(3, 1, 1), padding=(1, 0, 0))

        # self.diff = spconv.SparseSequential(
        #
        #     post_act_block(planes, planes, kernel_size=(1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1),
        #                    indice_key='subm1', conv_type='subm'),
        #     spconv.SparseAvgPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), indice_key='avgpool'),
        #     post_act_block(inplanes, inplanes, norm_fn=norm_fn, kernel_size=(1, 3, 3), stride=(1, 1, 1),
        #                    padding=(0, 0, 0), indice_key='avgpool', conv_type='inverseconv'),
        #     spconv.SparseConv3d(in_channels=planes, out_channels=2, kernel_size=(1, 1, 1), stride=1, padding=0))

        self.sed_layers = nn.ModuleList()
        for i in range(1):
            self.sed_layers.append(
                SEDLayer(32, [3, 3, 3], [2, 2, 2], [0, 0, 0], norm_fn=norm_fn, indice_key=f'sedlayer{i}'))
        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, stride[1], stride[1]),
                              padding=(1, 1, 1),
                              bias=False,
                              indice_key='spconv3'),
            norm_fn(planes),
            nn.ReLU(),
            spconv.SubMConv3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False,
                              indice_key='sumb3'),
            norm_fn(planes),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.conv1(x)
        temp = self.max_pool(x)
        indices = temp.indices.long()
        # x = replace_feature(x,x_expanded.features-x.features)

        # diff = self.diff(x)
        for conv in self.sed_layers:
            x = conv(x)

        # x = self.conv_out(x)

        x = spconv.SparseConvTensor(
            features=x.dense().permute(0, 2, 3, 4, 1)[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]],
            indices=temp.indices, spatial_shape=x.spatial_shape, batch_size=x.batch_size)
        return x


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
        x = Functional.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def sparse_concat(*tens: SparseConvTensor, dim):
    n_features = [ten.features.shape[-1] for ten in tens]
    n_features_all = sum(n_features)
    temp_index = 0
    results = []
    if dim == -1:
        for i in range(len(n_features)):
            temp_features = tens[i].features.new_zeros((tens[i].features.shape[0], n_features_all))
            temp_features[:, temp_index:(temp_index + n_features[i])] = tens[i].features
            results.append(tens[i].replace_feature(temp_features))
            temp_index += n_features[i]
        return sparse_add(*results)


def sparse_compress(ten: SparseConvTensor, dim):
    assert dim < len(ten.spatial_shape)
    coord = ten.indices
    features = ten.features
    sp_tensor = []
    if dim == 0:
        for i in range(coord[:, 1].max().item() + 1):
            indice = coord[:, 1] == i
            sp_tensor.append(SparseConvTensor(
                features=features[indice],
                indices=torch.concat([coord[indice][:, 0:1], coord[indice][:, 2:]], dim=1),
                batch_size=ten.batch_size,
                spatial_shape=ten.spatial_shape[1:]
            ))
        return sparse_concat(*sp_tensor, dim=-1)


class SpeedSampler(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.is_train = model_cfg.get('TRAIN', True)
        self.ckpt = model_cfg.get('CKPT', None)
        self.stride = model_cfg.STRIDE
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        # self.voxel_size = [voxel_size[i] * self.stride[i] for i in range(3)]
        # self.grid_size = [round(grid_size[i] / self.stride[i]) for i in range(3)]
        self.pillar_size = np.array(model_cfg.pillar_size)
        self.pillar_spatial_size = [round((point_cloud_range[3 + i] - point_cloud_range[i]) / model_cfg.pillar_size[i])
                                    for i in range(2)]
        self.use_norm = model_cfg.USE_NORM
        self.use_absoluto_xyz = self.model_cfg.get('USE_ABSOLUTE_XYZ', False)
        self.num_point_features = 12
        self.num_voxel_features = 36
        self.num_out_voxel_features = 32
        self.num_features_layers = [self.num_point_features] + list(model_cfg.NUM_FILTERS)
        self.num_features = self.num_out_voxel_features + self.num_features_layers[-1]

        self.name = 'speedsampler'
        self.point_cloud_range = point_cloud_range
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.num_voxel_features, out_features=self.num_out_voxel_features),
            nn.BatchNorm1d(self.num_out_voxel_features, eps=1e-3, momentum=0.01),
            nn.ReLU())
        from functools import partial
        self.gen = PointToVoxel(
            vsize_xyz=model_cfg.pillar_size[:-1] + [
                model_cfg.pillar_size[-1] / model_cfg.transform_points_to_pillars.NUM_FRAMES],
            coors_range_xyz=point_cloud_range,
            num_point_features=7,
            max_num_voxels=model_cfg.transform_points_to_pillars.MAX_NUMBER_OF_PILLARS * model_cfg.transform_points_to_pillars.NUM_FRAMES,
            max_num_points_per_voxel=model_cfg.transform_points_to_pillars.MAX_POINTS_PER_PILLAR,
            device=device
        )
        self.gen_voxel = PointToVoxel(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=6 + 2, max_num_voxels=150000,
            max_num_points_per_voxel=5,
            device=device
        )

        norm_fn = partial(nn.BatchNorm1d, momentum=0.01, eps=1e-3)
        self.sp_conv = TemperalDownConv(self.num_out_voxel_features, 32, stride=[2, 2],
                                        norm_fn=norm_fn,
                                        )

        self.compressed_conv = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, (3, 1, 1), padding=0, stride=1, indice_key='spconv'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.max_pool = spconv.SparseSequential(
            spconv.SparseMaxPool3d(kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), indice_key='pool_1'),
            post_act_block(32, 32, kernel_size=(5, 1, 1), norm_fn=norm_fn, padding=(1, 0, 0), stride=(2, 1, 1),
                           indice_key='pool_1',
                           conv_type='inverseconv'),

        )

        self.pool_3d_ = spconv.SparseMaxPool3d(kernel_size=3, stride=1, padding=1, indice_key='spconv')
        self.compressed_conv_1 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (3, 1, 1), padding=0, stride=1),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv2d = spconv.SparseSequential(
            spconv.SubMConv2d(64, 64, kernel_size=(3, 3), padding=1, stride=1, indice_key='conv2d_1'),
            norm_fn(64),
            nn.ReLU(),
            spconv.SubMConv2d(64, 64, kernel_size=(3, 3), padding=1, stride=1, indice_key='conv2d_1'),
            norm_fn(64),
            nn.ReLU()
        )
        # self.sp_up_conv = TemperalUpConv(32, 16, stride=[2, 2], norm_fn=norm_fn)
        pfn = []
        for i in range(len(self.num_features_layers) - 1):
            in_channels = self.num_features_layers[i]
            out_channels = self.num_features_layers[i + 1]
            pfn.append(PFNLayer(in_channels, out_channels, use_norm=self.use_norm,
                                last_layer=True if i == len(self.num_features_layers) - 2 else False))
        self.pfn = nn.ModuleList(pfn)
        # self.pool1 = spconv.SparseMaxPool2d(kernel_size=3,stride=1,padding=1,indice_key='conv2d_3')
        # self.pool2 = spconv.SparseMaxPool3d()
        # self.pool = spconv.SparseGlobalMaxPool()
        cur_num_features = self.num_features_layers[-1]
        out_num_features = [int(cur_num_features / 2), int(cur_num_features / 4), int(cur_num_features / 4)]
        # self.motion_conv = nn.Conv2d(in_channels=self.num_features * 2, out_channels=2, kernel_size=3)
        # self.motion_conv = spconv.SparseSequential(
        #     spconv.SubMConv3d(48 * 2, 64, kernel_size=(1, 3, 3)),
        #     norm_fn(64),
        #     nn.ReLU()
        # )

        # self.conv = nn.ModuleList([
        #     nn.ModuleList([nn.Sequential(
        #         nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
        #         nn.Conv2d(in_channels=cur_num_features * 2, out_channels=out_num_features[0],
        #                   kernel_size=2 * i + 3, stride=1),
        #         nn.BatchNorm2d(out_num_features[0], eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     ) for i in range(3)]),
        #     nn.ModuleList([nn.Sequential(
        #         nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
        #         nn.Conv2d(in_channels=out_num_features[0] * 3 * 2, out_channels=out_num_features[1],
        #                   kernel_size=2 * i + 3, stride=1),
        #         nn.BatchNorm2d(out_num_features[1], eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     ) for i in range(3)]),
        #     nn.ModuleList([nn.Sequential(
        #         nn.ZeroPad2d(padding=(i + 1, i + 1, i + 1, i + 1)),
        #         nn.Conv2d(in_channels=out_num_features[1] * 3 * 2, out_channels=out_num_features[2],
        #                   kernel_size=2 * i + 3, stride=1),
        #         nn.BatchNorm2d(out_num_features[2], eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     ) for i in range(3)])
        # ])

        # self.conv2d = spconv.SparseSequential(
        #     spconv.SparseConv2d(in_channels=16 * 3 * 2, out_channels=32, kernel_size=3, padding=1),
        #     norm_fn(32),
        #     nn.ReLU(),
        #
        #     spconv.SparseConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     norm_fn(32),
        #     nn.ReLU(),
        #
        #     spconv.SparseConv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #     norm_fn(16),
        #     nn.ReLU()
        # )

        # self.dense_conv2d = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )

        # self.regression_2d = spconv.SparseSequential(
        #
        #     spconv.SparseConv2d(16, 8, kernel_size=1,padding=0),
        #     norm_fn(8),
        #     nn.ReLU(),
        #     spconv.SparseConv2d(8, 2, kernel_size=1,padding=0)
        # )

        self.regression_2d = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=1, padding=0)
        )
        # self.classfier_2d = nn.Sequential(
        #     nn.Conv2d(16, 8, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 1, kernel_size=1, padding=0)
        # )
        self.classfier = spconv.SparseSequential(
            # spconv.SparseMaxPool2d(kernel_size=3,stride=1,padding=1,indice_key='pool_classifier'),
            # spconv.SparseInverseConv2d(in_channels=64,out_channels=64,kernel_size=3,indice_key='pool_classifier'),
            # norm_fn(64),
            # nn.ReLU(),
            spconv.SubMConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                              indice_key='classifier'),
            norm_fn(32),
            nn.ReLU(),
            spconv.SubMConv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.is_moving = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                              indice_key='is_moving'),
            norm_fn(32),
            nn.ReLU(),
            spconv.SubMConv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.regression = spconv.SparseSequential(
            # spconv.SparseMaxPool2d(kernel_size=3, stride=1, padding=1, indice_key='pool_classifier'),
            # spconv.SparseInverseConv2d(in_channels=64, out_channels=64, kernel_size=3, indice_key='pool_classifier'),
            # norm_fn(64),
            # nn.ReLU(),
            spconv.SubMConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                              indice_key='classifier'),
            norm_fn(32),
            nn.ReLU(),
            spconv.SubMConv1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

        # self.regression = spconv.SparseSequential(
        #     spconv.SubMConv2d(in_channels=16 * 3 * 2, out_channels=48, kernel_size=3, padding=1),
        #     norm_fn(48),
        #     nn.ReLU(),
        #     spconv.SubMConv2d(in_channels=48, out_channels=16, kernel_size=1),
        #     norm_fn(16),
        #     nn.ReLU(),
        #     spconv.SubMConv2d(in_channels=16, out_channels=2, kernel_size=1)
        # )
        self.sigmoid = nn.Sigmoid()

        self.pillar_x = self.pillar_size[0]
        self.pillar_y = self.pillar_size[1]
        self.pillar_z = self.pillar_size[2]
        self.offset_x = self.pillar_x / 2 + point_cloud_range[0]
        self.offset_y = self.pillar_y / 2 + point_cloud_range[1]
        self.offset_z = self.pillar_z / 2 + point_cloud_range[2]
        # self.embedding1 = nn.Sequential(
        #     nn.Conv1d(in_channels=9, out_channels=16, kernel_size=1),
        #     nn.BatchNorm1d(num_features=16),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        # self.diff_embedding1 = nn.Sequential(
        #     nn.Conv1d(in_channels=9, out_channels=16, kernel_size=1),
        #     nn.BatchNorm1d(num_features=16),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        #
        #
        #
        # self.diff_speed_pred = nn.Sequential(
        #     nn.Conv1d(in_channels=32 * 2, out_channels=16, kernel_size=1),
        #     nn.BatchNorm1d(num_features=16),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=16, out_channels=2, kernel_size=1)
        # )

    def decode_velocity(self, velocity):
        velocity[velocity > 0] = velocity[velocity > 0] ** 2
        velocity[velocity < 0] = -velocity[velocity < 0] ** 2
        return velocity

    def get_output_feature_dim(self):
        return self.num_point_features

    def get_clsloss(self, pre_speed):

        pass

    def transform_points_to_pillars(self, data_dict, config):

        voxels = []
        coors = []
        nums_points = []
        points = data_dict['points']
        if config.get('FILTER_GROUND', False) is not False:
            bin = torch.tensor([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]).to(device)

        B = data_dict['batch_size']
        F = data_dict['num_points_all'].shape[1]
        for b in range(B):
            voxels_single_batch = []
            coors_single_batch = []
            num_points_single_batch = []

            points_single_batch = points[points[:, 0] == b][:, 1:]
            if config.get('FILTER_GROUND', False) is not False:
                num = torch.tensor([torch.logical_and(points_single_batch[:, 2] >= bin[i],
                                                      points_single_batch[:, 2] < bin[i + 1]).sum() for i in
                                    range(len(bin) - 1)])
                GROUND = bin[torch.argmax(num) + 1] + config.MARGIN
            else:
                GROUND = -5
            voxels_single_batch, coors_single_batch, num_points_single_batch = self.gen(
                torch.concat([points_single_batch[:, :2],
                              points_single_batch[:, -1:] * self.gen.vsize[0] * 10 + self.gen.vsize[0] / 2 +
                              self.gen.coors_range[0], points_single_batch[:, 2:]], dim=-1)
            )

            voxels_single_batch = torch.concat([voxels_single_batch[:, :, :2], voxels_single_batch[:, :, 3:]], dim=-1)
            mask = (voxels_single_batch[:, :, 2].max(-1)[0] > GROUND)*((voxels_single_batch[:,:,2].max(-1)[0]-voxels_single_batch[:,:,2].min(-1)[0])>0.05)

            # for f in range(F):
            #     voxel,coor,num_points = self.gen(points_single_batch[points_single_batch[:,-1]==f*0.1][:,:-1])
            #     coor = torch.concat([torch.full((voxel.shape[0],1),f).to(device),coor],dim=-1)
            #     voxels_single_batch.append(voxel)
            #     coors_single_batch.append(coor)
            #     num_points_single_batch.append(num_points)

            coors_single_batch = torch.concat(
                [torch.full((coors_single_batch.shape[0], 1), b).to(device), coors_single_batch], dim=-1)

            voxels.append(voxels_single_batch[mask])
            coors.append(coors_single_batch[mask])
            nums_points.append(num_points_single_batch[mask])
            gt_boxes = data_dict['gt_boxes'][b]
            gt_boxes_mask = (gt_boxes[:,2]+gt_boxes[:,5]/2)<GROUND
            if (gt_boxes_mask).sum()>1:
                pass
        data_dict['pillars'] = torch.concat(voxels)
        data_dict['pillar_coords'] = torch.concat(coors)
        data_dict['pillar_num_points'] = torch.concat(nums_points)
        return data_dict

    def forward(self, batch_dict, train_box=True, **kwargs):
        # num_points = batch_dict['num_points_all'].int()

        frame_num = batch_dict['num_points_all'].shape[1]
        # frame_num = batch_dict['']
        # voxel_generator = VoxelGeneratorWrapper(
        #     vsize_xyz=self.voxel_size,
        #     coors_range_xyz=self.point_cloud_range,
        #     num_point_features=points_all.shape[1] - 1,
        #     max_num_points_per_voxel=self.model_cfg.MAX_POINTS_PER_VOXEL,
        #     max_num_voxels=self.model_cfg.MAX_NUMBER_OF_VOXELS['train' if self.training else 'test'],
        # )
        # voxels = []
        # coordinates = []
        # nums_points_voxels = []
        #
        B = batch_dict['batch_size']
        #

        # for batch in range(B):
        #     points = points_all[:, 1:][points_all[:, 0] == batch]
        #     for frame in range(frame_num):
        #         points_single_frame = points[points[:, -1] == frame / 10].to('cpu').numpy()
        #         voxel_output = voxel_generator.generate(points_single_frame)
        #         voxel, coordinate, num_points = voxel_output
        #         coordinate = np.concatenate(
        #             [batch * np.ones((coordinate.shape[0], 1)), frame * np.ones((coordinate.shape[0], 1)), coordinate],
        #             axis=1)
        #         voxels.append(torch.from_numpy(voxel).to(device))
        #         nums_points_voxels.append(torch.from_numpy(num_points).to(device))
        #         coordinates.append(torch.from_numpy(coordinate).to(device))
        #         num_points_all[batch, frame] = points_single_frame.shape[0]
        # nums_points=torch.tensor(nums_points_voxels).view((B,frame_num))
        # voxels = torch.concat(voxels)
        # coordinates = torch.concat(coordinates)
        # voxel_num_points = torch.concat(nums_points_voxels)

        batch_dict = self.transform_points_to_pillars(batch_dict, self.model_cfg.transform_points_to_pillars)
        voxel_features = batch_dict['pillars']
        voxels = voxel_features
        coordinates = batch_dict['pillar_coords']
        voxel_num_points = batch_dict['pillar_num_points']
        coordinates = torch.concat([coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 3:4], coordinates[:, 2:3]],
                                   dim=-1)  # (b,f,z,y,x) -> (b,f,x,y)

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coordinates[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.pillar_x + self.offset_x)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coordinates[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.pillar_y + self.offset_y)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coordinates[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.pillar_z + self.offset_z)
        features = [voxel_features, f_cluster, f_center]
        features = torch.concat(features, dim=-1)

        padding = torch.arange(voxel_features.shape[-2]).reshape((1, -1)).repeat(voxel_features.shape[0], 1).to(device)
        mask = padding < voxel_num_points.reshape((-1, 1))
        features[~mask] = 0
        for pfn in self.pfn:
            features = pfn(features)
        features = features.squeeze()

        # mean_height = voxels[:,:,2:3].mean(dim= -2)
        # size = voxels[:, :, :3].max(dim=-2)[0] - voxels[:, :, :3].min(dim=-2)[0]

        voxel_features = torch.concat([points_mean.squeeze(), voxel_num_points[:, None], features], dim=-1)
        voxel_features = self.linear(voxel_features)
        features = voxel_features

        B = int(coordinates[:, 0].max().item()) + 1
        # voxels[:, :, 5] *= 10
        F = int(coordinates[:, 1].max().item()) + 1
        H, W = self.pillar_spatial_size[:2]

        input_sp_tensor = spconv.SparseConvTensor(
            features=features,
            indices=coordinates.to(dtype=torch.int32),
            spatial_shape=[F, H + 1, W + 1],  # easy to implement deconv
            batch_size=B
        )

        sp_feature = self.sp_conv(input_sp_tensor)
        # sp_feature = self.compressed_conv(sp_features_list)
        # sp_feature = self.max_pool(sp_features_list)

        # sp_feature_padding = spconv.SparseConvTensor(
        #     features=sp_feature.features,
        #     indices=sp_feature.indices,
        #     spatial_shape=[F - 2 + 1, H, W],
        #     batch_size=B
        # )
        # sp_feature = self.compressed_conv_1(sp_feature_padding)
        sp_feature = spconv.SparseConvTensor(
            features=sp_feature.features,
            indices=torch.concat([sp_feature.indices[:, 0:1], sp_feature.indices[:, 2:]], dim=-1),
            spatial_shape=[H, W],
            batch_size=B
        )
        # sp_feature = self.conv2d(sp_feature)
        # spconv.SparseGlobalMaxPool
        is_gt_pred = self.classfier(sp_feature)
        is_moving_pred = self.is_moving(sp_feature)
        velocity_pred = self.regression(sp_feature)
        if not train_box:
            # batch_dict['diff_pred'] = diff.features
            batch_dict['is_gt_pred'] = is_gt_pred.features
            batch_dict['speed_pred'] = velocity_pred.features
            batch_dict['is_moving_pred'] = is_moving_pred.features
            batch_dict['coords_pred'] = is_gt_pred.indices
            batch_dict['coords_4frames_pred'] = input_sp_tensor.indices
            batch_dict['points'][:, 0] = batch_dict['points'][:, 0] * F + batch_dict['points'][:, -1] * 10
            batch_dict['batch_size'] *= F
            # batch_dict['voxel_coords'] = torch.concat(
            #     [batch_dict['voxel_coords'][:, :1] * frame_num + batch_dict['voxel_coords'][:, 1:2],
            #      batch_dict['voxel_coords'][:, 2:]], dim=-1)
        else:

            self.gen_voxel_full = PointToVoxel(
                vsize_xyz=[1] + self.voxel_size,
                coors_range_xyz=np.concatenate(
                    [np.array([-0.5]), self.point_cloud_range[:3], np.array([B - 0.5]), self.point_cloud_range[3:]],
                    axis=0),
                num_point_features=6 + (3 if self.model_cfg.TRAIN_WITH_VEL else 1), max_num_voxels=150000 * B,
                max_num_points_per_voxel=5,
                device=device
            )
            points = batch_dict['points']
            is_gt_pred = replace_feature(is_gt_pred, self.sigmoid(is_gt_pred.features))
            is_gt_pred = is_gt_pred.dense().permute(0, 2, 3, 1).squeeze(-1) 
            is_moving_pred = replace_feature(is_moving_pred, self.sigmoid(is_moving_pred.features))
            is_moving_pred = is_moving_pred.dense().permute(0, 2, 3, 1).squeeze(-1) 

            velocity_pred = velocity_pred.dense().permute(0, 2, 3, 1)
            points_coords = torch.concat([points[:, 0:1], (points[:, 1:2] - self.point_cloud_range[0]) / self.pillar_x,
                                          (points[:, 2:3] - self.point_cloud_range[1]) / self.pillar_y], dim=-1).long()
            points_coords = torch.clamp((points_coords), min=0, max=H - 1)
            points_gt_mask = is_gt_pred[points_coords[:, 0], points_coords[:, 1], points_coords[:, 2]]>0.1
            points_gt_mask[points[:, -1] == 0] = True
            # points_gt_mask[points[:, -1] != 0] = False
            points = points[points_gt_mask]
            points_coords = points_coords[points_gt_mask]

            is_moving_pred = is_moving_pred[points_coords[:, 0], points_coords[:, 1], points_coords[:, 2]]>0.8
            velocity_pred = velocity_pred[points_coords[:, 0], points_coords[:, 1], points_coords[:, 2]]
            velocity_pred = self.decode_velocity(velocity_pred)
            velocity_pred[~is_moving_pred] = 0
            points[:, 1:3][is_moving_pred] += velocity_pred[is_moving_pred] * points[is_moving_pred][:, -1][:, None]
            if self.model_cfg.TRAIN_WITH_VEL:
                points = torch.concat([points, velocity_pred], dim=-1)

            if not self.model_cfg.TRAIN_WITH_VEL:
                batch_dict['gt_boxes'] = torch.concat(
                    [batch_dict['gt_boxes'][:, :, :7], batch_dict['gt_boxes'][:, :, -1:]], dim=-1)

                # voxels,voxel_coords,voxel_num_points = self.gen_voxel(points_single_batch)
            voxels_full, voxel_coords_full, voxel_num_points_full = self.gen_voxel_full(points)
            voxels_full = voxels_full[:, :, 1:]
            voxel_coords_full = torch.concat([voxel_coords_full[:, -1:], voxel_coords_full[:, :3]], dim=-1)
            # voxels = torch.concat([torch.full((voxels.shape[0],1),b).to(device),voxels],dim=-1)

            # voxels_list.append(voxels)
            # voxel_coordinates_list.append(voxel_coords)
            # voxel_num_points_list.append(voxel_num_points)

            batch_dict['voxels'] = voxels_full
            batch_dict['voxel_coords'] = voxel_coords_full
            batch_dict['voxel_num_points'] = voxel_num_points_full

            batch_dict['points'] = points

        return batch_dict
        bbox = True

        # motion_features = self.sp_up_conv(sp_features_list)
        # motion_features = sparse_compress(motion_features, dim=0)
        # if signal:
        #     motion_features = motion_features_list[0]
        #     cur_motion_mask = motion_features.indices[:, 1] < 2
        #     pre_motion_mask = motion_features.indices[:, 1] > 0
        #     aggregated_features, est_speed = self.attention_aggre(motion_features.features[cur_motion_mask],
        #                                                           motion_features.features[pre_motion_mask],
        #                                                           motion_features.indices[cur_motion_mask],
        #                                                           motion_features.indices[pre_motion_mask],
        #                                                           spatial_shape=torch.tensor(
        #                                                               [motion_features.spatial_shape[-2],
        #                                                                motion_features.spatial_shape[-1]]).to(device))
        # else:
        #     aggregated_features = motion_features

        if not self.training:
            # is_moving_mask = is_moving>0.5
            coordinate_1st_mask = coordinates[:, 1] > 0
            # coordinate_2st_mask = (coordinates[:, 1] < F - 1)
            coordinate_all = coordinates
        else:
            coordinate_all = coordinates
            coordinate_1st_mask = coordinates[:, 1] > 0
            # coordinate_2st_mask = coordinates[:,1]<F-1
        coordinate_1st = coordinates[coordinate_1st_mask]
        if bbox:
            # motion_features = motion_features.dense()
            # is_moving_pred = self.classfier(motion_features).dense().permute(0,2,3,1)[:,:-1,:-1,0]
            # is_moving_pred = is_moving_pred.reshape(B,is_moving_pred.shape[1],is_moving_pred.shape[2])
            # is_moving_pred = is_moving_pred[coordinates[:, 0], coordinates[:, 2], coordinates[:, 3]].squeeze()

            motion_features = self.conv2d(motion_features).dense()
            motion_features = self.dense_conv2d(motion_features)[:, :, :-1, :-1]
            speed = self.regression_2d(motion_features)
            is_moving_pred = self.classfier_2d(motion_features)
            batch_dict['speed_map_pred'] = speed.permute(0, 2, 3, 1)

            batch_dict['pillar_coords'] = coordinate_1st
            batch_dict['speed_1st'] = None
            batch_dict['is_moving_pred'] = is_moving_pred.permute(0, 2, 3, 1)[:, :, :, 0]

            batch_dict['points'][:, 0] = batch_dict['points'][:, 0] * F + batch_dict['points'][:, -1] * 10
            batch_dict['batch_size'] *= F
            batch_dict['voxel_coords'] = torch.concat(
                [batch_dict['voxel_coords'][:, :1] * frame_num + batch_dict['voxel_coords'][:, 1:2],
                 batch_dict['voxel_coords'][:, 2:]], dim=-1)

            return batch_dict
        classification_sp_tensor = self.classfier(aggregated_features)
        classification = self.sigmoid(classification_sp_tensor.dense()).permute(0, 2, 3, 1)

        speed_sp_tensor = self.regression(aggregated_features)
        speed = speed_sp_tensor.dense().permute(0, 2, 3, 1)
        batch_dict['speed_map_pred'] = speed[:, :-1, :-1, :]
        speed_new = speed.detach()

        is_moving = classification
        is_moving = is_moving[coordinates[:, 0], coordinates[:, 2], coordinates[:, 3]].squeeze()

        # coordinate_2st = coordinates[coordinate_2st_mask]
        num_voxel_1st = [torch.unique(coordinate_1st[coordinate_1st[:, 0] == b][:, 1], return_counts=True)[1] for b in
                         range(B)]
        # num_voxel_2st = [torch.unique(coordinate_2st[coordinate_2st[:, 0] == b][:, 1], return_counts=True)[1] for b in
        #                  range(B)]

        speed_1st = speed_new[coordinate_1st[:, 0], coordinate_1st[:, 2], coordinate_1st[:, 3]]
        # speed_2st = speed_new[coordinate_2st[:, 0], coordinate_2st[:, 2], coordinate_2st[:, 3]]
        voxels_1st = voxels[coordinate_1st_mask]
        # voxels_2st = voxels[coordinate_2st_mask]
        n_point_1st = voxel_num_points[coordinate_1st_mask]
        # n_point_2st = voxel_num_points[coordinate_2st_mask]

        proxy_points_1st = torch.zeros(voxels_1st.shape[0], 3).to(device)
        # proxy_points_2st = torch.zeros(voxels_2st.shape[0], 3).to(device)

        proxy_points_1st[:, :2] = coordinate_1st[:, 2:] * torch.tensor(self.voxel_size[:2])[None, :].to(
            device) + torch.tensor(
            self.point_cloud_range[:2])[None, :].to(device) + (torch.tensor(self.voxel_size[:2]) / 2)[None, :].to(
            device)
        # proxy_points_2st[:, :2] = coordinate_2st[:, 2:] * torch.tensor(self.voxel_size[:2])[None, :].to(
        #     device) + torch.tensor(
        #     self.point_cloud_range[:2])[None, :].to(device) + (torch.tensor(self.voxel_size[:2]) / 2)[None, :].to(
        #     device)

        proxy_points_1st[:, 2] = voxels_1st[:, :, 2].sum(dim=-1) / n_point_1st
        # proxy_points_2st[:, 2] = voxels_2st[:, :, 2].sum(dim=-1) / n_point_2st
        xyz = points_all[points_all[:, -1] > 0][:, 1:]

        num_points_all = [[(torch.logical_and(points_all[:, -1] == f / 10, points_all[:, 0] == b)).sum().item() for f in
                           range(frame_num)] for b in range(B)]
        num_points_all = torch.tensor(num_points_all).cuda()
        idx, empty_ball_mask = ball_query(0.5, 8, xyz[:, :3].contiguous(),
                                          num_points_all[:, 1:].reshape(-1).int().contiguous(),
                                          proxy_points_1st.contiguous(),
                                          torch.concat(num_voxel_1st).int().contiguous())

        grouped_feature_1st = grouping_operation(xyz.contiguous(), num_points_all[:, 1:].reshape(-1).int().contiguous(),
                                                 idx.contiguous(), torch.concat(num_voxel_1st).int().contiguous())
        grouped_feature_1st = torch.concat(

            [grouped_feature_1st, grouped_feature_1st[:, :3, :] - proxy_points_1st[:, :, None]], dim=1)

        # xyz = points_all[points_all[:, -1] < (F - 1) * 0.1][:, 1:]
        # idx, empty_ball_mask = ball_query(0.5, 8, xyz[:, :3].contiguous(),
        #                                   num_points_all[:, :-1].reshape(-1).int().contiguous(),
        #                                   proxy_points_2st.contiguous(),
        #                                   torch.concat(num_voxel_2st).int().contiguous())
        # grouped_feature_2st = grouping_operation(xyz.contiguous(),
        #                                          num_points_all[:, :-1].reshape(-1).int().contiguous(),
        #                                          idx.contiguous(), torch.concat(num_voxel_2st).int().contiguous())
        # grouped_feature_2st = torch.concat(
        #     [grouped_feature_2st, grouped_feature_2st[:, :3, :] - proxy_points_2st[:, :, None]], dim=1)
        refer_points_1st = torch.zeros(voxels_1st.shape[0], 3).to(device)
        # refer_points_2st = torch.zeros(voxels_2st.shape[0], 3).to(device)
        refer_points_1st[:, :2] = coordinate_1st[:, 2:] * torch.tensor(self.voxel_size[:2])[None, :].to(
            device) + torch.tensor(
            self.point_cloud_range[:2])[None, :].to(device) + (torch.tensor(self.voxel_size[:2]) / 2)[None, :].to(
            device) + speed_1st
        # refer_points_2st[:, :2] = coordinate_2st[:, 2:] * torch.tensor(self.voxel_size[:2])[None, :].to(
        #     device) + torch.tensor(
        #     self.point_cloud_range[:2])[None, :].to(device) + (torch.tensor(self.voxel_size[:2]) / 2)[None, :].to(
        #     device) + speed_2st

        refer_points_1st[:, 2] = voxels_1st[:, :, 2].sum(dim=-1) / n_point_1st
        # refer_points_2st[:, 2] = voxels_2st[:, :, 2].sum(dim=-1) / n_point_2st
        xyz = points_all[points_all[:, -1] < (F - 1) * 0.1][:, 1:]
        idx, empty_ball_mask = ball_query(1, 8, xyz[:, :3].contiguous(),
                                          num_points_all[:, :-1].reshape(-1).int().contiguous(),
                                          refer_points_1st.contiguous(),
                                          torch.concat(num_voxel_1st).int().contiguous())
        diff_grouped_feature_1st = grouping_operation(xyz.contiguous(),
                                                      num_points_all[:, :-1].reshape(-1).int().contiguous(),
                                                      idx, torch.concat(num_voxel_1st).int().contiguous())

        diff_grouped_feature_1st = torch.concat(
            [diff_grouped_feature_1st, diff_grouped_feature_1st[:, :3] - refer_points_1st[:, :, None]],
            dim=1)
        # xyz = points_all[points_all[:, -1] > 0][:, 1:]
        # idx, empty_ball_mask = ball_query(1, 8, xyz[:, :3].contiguous(),
        #                                   num_points_all[:, :-1].reshape(-1).int().contiguous(),
        #                                   refer_points_2st.contiguous(),
        #                                   torch.concat(num_voxel_2st).int().contiguous())
        # diff_grouped_feature_2st = grouping_operation(xyz.contiguous(),
        #                                               num_points_all[:, 1:].reshape(-1).int().contiguous(),
        #                                               idx, torch.concat(num_voxel_2st).int().contiguous())
        # diff_grouped_feature_2st = torch.concat(
        #     [diff_grouped_feature_2st, diff_grouped_feature_2st[:, :3] - refer_points_2st[:, :, None]],
        #     dim=1)

        grouped_feature_1st = self.embedding1(grouped_feature_1st)
        diff_grouped_feature_1st = self.diff_embedding1(diff_grouped_feature_1st)

        # grouped_feature_2st = self.embedding2(grouped_feature_2st)
        # diff_grouped_feature_2st = self.diff_embedding2(diff_grouped_feature_2st
        #                                                 )
        grouped_feature_1st = Functional.avg_pool1d(grouped_feature_1st,
                                                    kernel_size=[grouped_feature_1st.shape[-1]]).squeeze()
        # grouped_feature_2st = Functional.avg_pool1d(grouped_feature_2st,
        #                                             kernel_size=[diff_grouped_feature_2st.shape[-1]]).squeeze()

        diff_grouped_feature_1st = Functional.avg_pool1d(diff_grouped_feature_1st,
                                                         kernel_size=[diff_grouped_feature_1st.shape[-1]]).squeeze()
        # diff_grouped_feature_2st = Functional.avg_pool1d(diff_grouped_feature_2st,
        #                                                  kernel_size=[diff_grouped_feature_2st.shape[-1]]).squeeze()

        grouped_feature_1st = torch.concat([grouped_feature_1st - diff_grouped_feature_1st, grouped_feature_1st],
                                           dim=-1)
        # grouped_feature_2st = torch.concat([grouped_feature_2st - diff_grouped_feature_2st, grouped_feature_2st],
        #                                    dim=-1)

        diff_speed_1st = self.diff_speed_pred(grouped_feature_1st.unsqueeze(-1))
        # diff_speed_2st = self.diff_speed_pred(grouped_feature_2st.unsqueeze(-1))
        speed_1st = speed_1st + diff_speed_1st.squeeze()
        # speed_2st = speed_2st + diff_speed_2st.squeeze()
        speed_all = torch.zeros((coordinate_1st.shape[0], 2)).to(device)
        # speed_all[coordinate_all[:, 1] == F - 1] = speed_1st[coordinate_1st[:, 1] == F - 1]
        # # speed_all[coordinate_all[:, 1] == 0] = speed_2st[coordinate_2st[:, 1] == 0]
        # speed_all[(coordinate_all[:, 1] > 0) & (coordinate_all[:, 1] < F - 1)] = \
        # (speed_1st[(coordinate_1st[:, 1] > 0) & (coordinate_1st[:, 1] < F - 1)]+
        # speed_2st[(coordinate_2st[:, 1] > 0) & (coordinate_2st[:, 1] < F - 1)])/2
        speed_all[:] = speed_1st[:]
        batch_dict['pillar_coords'] = coordinate_1st
        batch_dict['speed_1st'] = speed_all
        batch_dict['is_moving_pred'] = is_moving[coordinate_1st_mask]

        batch_dict['points'][:, 0] = batch_dict['points'][:, 0] * F + batch_dict['points'][:, -1] * 10
        batch_dict['batch_size'] *= F
        batch_dict['voxel_coords'] = torch.concat(
            [batch_dict['voxel_coords'][:, :1] * frame_num + batch_dict['voxel_coords'][:, 1:2],
             batch_dict['voxel_coords'][:, 2:]], dim=-1)
        return batch_dict
        grouped_feature[:, :3] -= proxy_points[:, None, :]

        pillar_coords = motion_features.indices[cur_motion_mask]

        anti_mask = pillar_coords[:, 0] >= B
        pillar_coords[anti_mask, 1] = F - 1 - pillar_coords[anti_mask][:, 1]
        pillar_coords[anti_mask, 0] -= B
        pillar_coords[:, 0] = pillar_coords[:, 0] * F + pillar_coords[:, 1]
        pillar_coords = pillar_coords[:, [0, 2, 3]]

        batch_dict['speed_pred'] = est_speed
        batch_dict['pillar_coords'] = pillar_coords
        batch_dict['classification'] = classification

        num_points = torch.cumsum(batch_dict['num_points'].flatten(), dim=0).long()
        num_voxels = torch.cumsum(batch_dict['num_voxels'].flatten(), dim=0).long()
        num_gt_boxes = batch_dict['num_gt_boxes'].flatten().long()
        sum_gt_boxes = torch.cumsum(batch_dict['num_gt_boxes'], dim=-1).long()
        gt_boxes = torch.zeros((len(num_gt_boxes.flatten()), batch_dict['num_gt_boxes'].max().int().item(),
                                batch_dict['gt_boxes'].shape[-1])).to(device)
        for i in range(len(num_points)):
            batch_dict['points'][:, 0][(num_points[i - 1] if i > 0 else 0):num_points[i]] = i
        batch_dict['voxel_coords'][:, 0][(num_voxels[i - 1] if i > 0 else 0):num_points[i]] = i
        gt_boxes[i][:num_gt_boxes[i]] = batch_dict['gt_boxes'][i // F][
                                        (sum_gt_boxes[i // F][i % F - 1] if i % F else 0): sum_gt_boxes[i // F][
                                            i % F]]
        batch_dict['gt_boxes'] = gt_boxes
        batch_dict['batch_size'] = num_points.size()[0]
        return batch_dict

        speed_mask = torch.zeros((B, F + 1, H, W))

        speed_mask[coordinates[:, 0], coordinates[:, 1] + 1, coordinates[:, 2], coordinates[:, 3]] = 1
        speed_mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = 1
        speed_mask = speed_mask[:, 1:-1, :, :]

        pillar_map = torch.zeros(B, F, H, W, features.shape[-1]).to(device)
        pillar_map[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = features
        pillar_map = torch.permute(pillar_map, (0, 1, 4, 2, 3))
        cur_features = pillar_map[:, :-1, :, :, :].reshape((-1, self.num_features, H, W))
        pre_features = pillar_map[:, 1:, :, :, :].reshape((-1, self.num_features, H, W))
        cur_features = features[coordinates[:, 1] < F - 1]
        pre_features = features[coordinates[:, 1] > 0]
        cur_voxels = voxels[coordinates[:, 1] < F - 1]
        pre_voxels = voxels[coordinates[:, 1] > 0]

        cur_coordinates = coordinates[coordinates[:, 1] < F - 1]
        pre_coordinates = coordinates[coordinates[:, 1] > 0]
        num_match = 12
        match_coord = torch.zeros((pre_coordinates.shape[0], num_match)).long()

        index = 0
        for b in range(B):
            for f in range(F - 1):
                cur = cur_coordinates[(cur_coordinates[:, 0] == b) & (cur_coordinates[:, 1] == f)][:, -2:]
                pre = pre_coordinates[(pre_coordinates[:, 0] == b) & (pre_coordinates[:, 1] == f + 1)][:, -2:]
                distances = torch.norm(pre.unsqueeze(1) - cur.float(), dim=-1)
                _, indices = torch.topk(distances, k=num_match, dim=-1, largest=False)
                match_coord[index:index + pre.shape[0]] = indices
                index += pre.shape[0]
        match_features = cur_features[match_coord]
        match_atten = torch.mul(pre_features.unsqueeze(1), match_features).sum(dim=-1)

        max_locc = match_coord[torch.arange(match_coord.shape[0]), match_atten.max(dim=-1)[1]]

        pred_speed = cur_voxels[max_locc].mean(dim=-2)[:, :2] - pre_voxels.mean(dim=-2)[:, :2]

        batch_dict['pred_speed'] = pred_speed
        return batch_dict
