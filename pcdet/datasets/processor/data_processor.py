from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils
from pcdet import device
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                # from spconv.utils import Point2VoxelGPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        self.pillar_generator = None
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1),
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)

        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features + (1 if data_dict.get('label', False) is not False else 0),
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
        points = data_dict['points']
        if not config.get('CONCAT', True):
            voxels = []
            coordinates = []
            num_points = []

            frame_num = data_dict['num_points_all'].shape[0]
            num_voxels = np.zeros((frame_num)).astype(np.int64)
            for frame in range(data_dict['num_points_all'].shape[0]):
                voxel_output = self.voxel_generator.generate(points[points[:, -1] == 0.1 * frame,:-1])
                voxels.append(voxel_output[0])
                coordinate = np.concatenate([np.ones([voxel_output[1].shape[0],1],dtype=np.int32)*frame,voxel_output[1]],axis=1)
                coordinates.append(coordinate)
                num_points.append(voxel_output[2])
                num_voxels[frame] = voxel_output[0].shape[0]
            data_dict['num_voxels'] = num_voxels
            voxels = np.concatenate(voxels)
            coordinates = np.concatenate(coordinates)
            num_points = np.concatenate(num_points)
        else:
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            voxels = voxels[:,:,:-1] if config.get('REMOVE_TIME_STAMP',False) else voxels[:,:,:]
        # if config.get('POINT_FEATURES', None) is not None:
        #
        #
        #     L, W, H = self.grid_size
        #     dense_voxel = torch.zeros((L, W, H, voxels.shape[-1] * voxels.shape[-2]))
        #     split = [coordinates[:, -i - 1] for i in range(coordinates.shape[1])]
        #     dense_voxel[split] = torch.from_numpy(voxels.reshape(voxels.shape[0], -1))
        #
        #     H, W, D, C = dense_voxel.size()
        #     dense_voxel = dense_voxel.reshape((H * W, D * voxels.shape[-2], -1))
        #
        #     points_num_pillar = ((dense_voxel[:, :, 1] != 0) + (dense_voxel[:, :, 2] != 0) + (
        #             dense_voxel[:, :, 3] != 0)).sum(axis=-1)
        #     dense_voxel = dense_voxel[points_num_pillar != 0]
        #     points_num_pillar = points_num_pillar[points_num_pillar != 0]
        #
        #     point_features = [dense_voxel[:, :, :-1]]
        #     num_point_features = self.num_point_features
        #     if 'mean_z' in config['POINT_FEATURES']:
        #         mean_z = torch.sum(dense_voxel[:, :, 2], dim=-1) / points_num_pillar
        #         mean_z = mean_z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        #         point_features.append(mean_z.expand(-1, dense_voxel.shape[1], -1))
        #         num_point_features += 1
        #     if 'height' in config['POINT_FEATURES']:
        #         height = torch.max(dense_voxel[:, :, 2], dim=-1)[0] - \
        #                  torch.min(dense_voxel[:, :, 2], dim=-1)[0]
        #         height = height.unsqueeze(dim=-1).unsqueeze(dim=-1)
        #         point_features.append(height.expand(-1, dense_voxel.shape[1], -1))
        #         num_point_features += 1
        #
        #     voxel_generator = VoxelGeneratorWrapper(
        #         vsize_xyz=config.VOXEL_SIZE,
        #         coors_range_xyz=self.point_cloud_range,
        #         num_point_features=num_point_features,
        #         max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
        #         max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
        #     )
        #
        #     point_features.append(dense_voxel[:, :, -1].unsqueeze(-1))
        #     point_features = np.concatenate(point_features, axis=-1)
        #     point_features = point_features.reshape(-1, point_features.shape[-1])
        #     point_features = point_features[
        #         (point_features[:, 1] != 0) * (point_features[:, 2] != 0) * (point_features[:, 3] != 0)]
        #
        #     voxel_output = voxel_generator.generate(point_features)
        #     voxels, coordinates, num_points = voxel_output
        #     data_dict['points'] = point_features

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = np.concatenate(voxels_list)
            data_dict['voxel_coords'] = np.concatenate(voxel_coords_list)
            data_dict['voxel_num_points'] = np.concatenate(voxel_num_points_list)

        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points

        return data_dict


    def select_trajectory_boxes(self,data_dict=None,config=None):
        
        
        def select_boxes(gt_boxes,dis_threshold):
            
            gt_tra_boxes = [[] for i in range(FRAME)]
            gt_boxes[0][:, -3] = np.arange(gt_boxes[0].shape[0]) + 1

            idx = gt_boxes[0].shape[0] + 1
            for i in range(FRAME - 1):
                cur_boxes = np.concatenate(gt_boxes[:i + 1])
                pre_boxes = gt_boxes[i + 1]
                vel_pre_abs = abs(pre_boxes[:, 7:9])
                dis = np.sqrt(((((cur_boxes[:, :2] - 0.1 * (i + 1 - cur_boxes[:, -2:-1]) * cur_boxes[:, 7:9])[None, :,
                                 :] - pre_boxes[:, :2][:, None, :]) / np.clip(vel_pre_abs, a_min=1, a_max=None)[:, None,
                                                                      :]) ** 2).sum(axis=-1))

                vel_dis = np.sqrt((((cur_boxes[:, 7:9][None, :, :] - pre_boxes[:, 7:9][:, None, :]) / np.clip(
                    vel_pre_abs, a_min=1, a_max=None)[:, None, :]) ** 2).sum(axis=-1))

                dis += vel_dis
                if dis.shape[0] * dis.shape[1] == 0:
                    continue
                dis_min = np.min(dis, axis=-1)
                arg_min = np.argmin(dis, axis=-1)

                pre_boxes[:, -3][dis_min < dis_threshold] = cur_boxes[:, -3][arg_min][dis_min < dis_threshold]
                pre_boxes[:, -3][dis_min >= dis_threshold] = np.arange(idx, idx + (dis_min >= dis_threshold).sum())
                idx += (dis_min >= dis_threshold).sum()
            return idx,gt_boxes


        if data_dict is None:
            return partial(self.select_trajectory_boxes,config=config)
        
        FRAME = data_dict['num_points_all'].shape[0]
        dis_threshold = config.DIS_THRE
        
        gt_boxes = data_dict['gt_boxes']
        gt_boxes = np.concatenate([gt_boxes[:, :-2], np.zeros([gt_boxes.shape[0], 1]), gt_boxes[:, -2:]], axis=-1)
        gt_boxes = [gt_boxes[gt_boxes[:, -2] == i + 1] for i in range(FRAME)]
        
        idx, gt_boxes = select_boxes(gt_boxes, dis_threshold=config.DIS_THRE)
        
        try:
            assert (idx-max([gt_box.shape[0] for gt_box in gt_boxes]))<8
        except:
            pass
        if (idx-max([gt_box.shape[0] for gt_box in gt_boxes])) > max(gt_boxes[0].shape[0]/2,5): ## avoid vibrate diff
            idx,gt_boxes = select_boxes(gt_boxes,dis_threshold = config.DIS_THRE2)
            if (idx - max([gt_box.shape[0] for gt_box in gt_boxes])) > max(gt_boxes[0].shape[0] / 3,
                                                                           5):  ## avoid vibrate diff
                idx, gt_boxes = select_boxes(gt_boxes, dis_threshold=config.DIS_THRE2+3)
        
        data_dict['gt_boxes'] = np.concatenate(gt_boxes,axis=0)
        return data_dict

    def transform_points_to_pillars(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.PILLAR_SIZE)
            self.speed_grid_size = np.round(grid_size).astype(np.int64)
            self.pillar_size = config.PILLAR_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_pillars, config=config)
        num_point_features = self.num_point_features + (1 if data_dict.get('label', False) is not False else 0)+(1 if config.get('WITH_TIME_STAMP',False) else 0)
        
        if self.pillar_generator is None:
            self.pillar_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.PILLAR_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_PILLAR,
                max_num_voxels=config.MAX_NUMBER_OF_PILLARS[self.mode],
            )
        points = data_dict['points']
        if not config.get('CONCAT', True):
            pillars = []
            coordinates = []
            num_points = []
            frame_num = data_dict['num_points_all'].shape[0]
            num_pillars = np.zeros((frame_num)).astype(np.int64)
            gt_boxes = data_dict['gt_boxes']
            if config.get('FILTER_GROUND', False) is not False:
                bin = np.array([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
                num,_ = np.histogram(points[:,2],bin)
                GROUND = bin[np.argmax(num)+1]+config.MARGIN
                points = points[points[:,2]>bin[np.argmax(num)+1]+config.MARGIN]
                gt_mask = data_dict['gt_boxes'][:, 2] >= GROUND
                data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_mask]
                data_dict['gt_names'] = data_dict['gt_names'][gt_mask]
            for frame in range(frame_num):
                points_single_frame = points[points[:, -1] == 0.1 * frame, :-1]
                ratio = config.get('RM_BK_RATIO',None)
                if ratio is not None:
                    ratio = config.RM_BK_RATIO[self.mode]
                    gt_box_single_frame = gt_boxes[gt_boxes[:,-2]==frame][:,:7]

                    points1,points2= np.split(points_single_frame,np.array([points_single_frame.shape[0]*ratio],dtype=np.int))
                    box_point_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(points1[:, 0:3]).float(),
                        torch.from_numpy(gt_box_single_frame[:, 0:7]).float()
                    ).long().numpy()

                    box_point_mask = np.sum(box_point_mask,axis=0)>0
                    points_single_frame = np.concatenate([points1[box_point_mask],points2],axis=0)
                pillar_output = self.pillar_generator.generate(points_single_frame)
                if config.get('WITH_TIME_STAMP',False):
                    pillars.append(np.concatenate([pillar_output[0],np.ones([pillar_output[0].shape[0],pillar_output[0].shape[1],1])*0.1*frame],axis=-1))
                else:
                    pillars.append(pillar_output[0])

                coordinate = np.concatenate([np.ones([pillar_output[1].shape[0],1],dtype=np.int32) * frame,pillar_output[1]],axis=-1)
                coordinates.append(coordinate)
                num_points.append(pillar_output[2])
                num_pillars[frame] = pillar_output[0].shape[0]

            data_dict['num_pillars'] = num_pillars
            pillars = np.concatenate(pillars)
            coordinates = np.concatenate(coordinates)
            num_points = np.concatenate(num_points)
        else:
            pillar_output = self.pillar_generator.generate(points)
            pillars, coordinates, num_points = pillar_output

        if not data_dict['use_lead_xyz']:
            pillars = pillars[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            pillars_list, pillar_coords_list, pillar_num_points_list = [pillars], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                pillar_output = self.pillar_generator.generate(points_list[i])
                pillars, coordinates, num_points = pillar_output

                if not data_dict['use_lead_xyz']:
                    pillars = pillars[..., 3:]
                pillars_list.append(pillars)
                pillar_coords_list.append(coordinates)
                pillar_num_points_list.append(num_points)

            data_dict['pillars'] = np.concatenate(pillars_list)
            data_dict['pillar_coords'] = np.concatenate(pillar_coords_list)
            data_dict['pillar_num_points'] = np.concatenate(pillar_num_points_list)

        else:
            data_dict['pillars'] = pillars
            data_dict['pillar_coords'] = coordinates
            data_dict['pillar_num_points'] = num_points

        return data_dict
    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict

    def image_calibrate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            if cur_processor.keywords['config'].NAME=='transform_points_to_voxels' and cur_processor.keywords['config'].get('GPU',False):
                continue
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
