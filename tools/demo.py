import argparse
import glob
import os.path
from pathlib import Path
import pickle

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
    import mayavi.mlab as mlab
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/centerpoint_4frames.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str,
                        default='/media/msun/Seagate/waymo/waymo_processed_data_v0_5_0/segment-1005081002024129653_5313_150_5333_150_with_camera_labels/0001.npy',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str,
                        default='../output/waymo_models/centerpoint_4frames/default/ckpt/latest_model.pth',
                        help='specify the pretrained model')
    parser.add_argument('--func', type=str, default='visualize_file', help='func you want to excecute')
    parser.add_argument('--dataset', type=str, default='waymo', help='dataset you want to visualize')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    parser.add_argument('--frame_rate', type=int, default=100, help='frame_rate of auto-displaying')
    parser.add_argument('--auto_display', type=bool, default=False, help='whether to display automately')
    parser.add_argument('--index', type=int, default=9, help='index of data in your dataset')
    parser.add_argument('--frames', type=int, default=1, help='frames to cat')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


def show():
    # data_path = '../data/waymo/waymo_processed_data_v0_5_0/segment-272435602399417322_2884_130_2904_130_with_camera_labels'
    # #data_path = '/media/msun/Seagate/radraed_lidar_data/infared-lidar/pro_data/lidar_matched/2023-10-29-15-18-18/2052.pcd'
    # #data_path = '/media/msun/Seagate/waymo/waymo_processed_data_v0_5_0/segment-272435602399417322_2884_130_2904_130_with_camera_labels/0001.npy'
    # data_path = '/media/msun/Seagate/nuscenes/v1.0-mini/raw_data/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801003948191.npy'
    args, cfg = parse_config()
    data_path = args.data_path
    if os.path.isfile(data_path):
        if data_path.endswith('.npy'):
            V.draw_scenes(
                points=np.load(data_path),
            )
        if data_path.endswith('.pcd'):
            V.draw_pcd(data_path)
        if data_path.endswith('.bin'):
            points = np.fromfile(data_path, dtype=np.float32).reshape((-1, 4))
            V.draw_scenes(points=points[:, :3], file_name=data_path)

        if not OPEN3D_FLAG:
            mlab.show(stop=True)
    else:
        files = os.listdir(data_path)
        frames = []
        file_names = []
        if not args.auto_display:
            for file in files:
                if file.endswith('.npy'):
                    points = np.load(os.path.join(data_path, file))
                    V.draw_scenes(points=points)
                elif file.endswith('.bin'):
                    with open(os.path.join(data_path, file), 'rb') as file_name:
                        points = np.fromfile(file_name, dtype=np.float32)
                    V.draw_scenes(points=points.reshape((-1, 4))[:, :3], file_name=file)
                if file.endswith('.pcd'):
                    V.draw_pcd(data_path)
        else:
            for file in files:
                if file.endswith('.npy'):
                    points = np.load(os.path.join(data_path, file))
                elif file.endswith('.bin'):
                    with open(os.path.join(data_path, file), 'rb') as file_name:
                        points = np.fromfile(file_name, dtype=np.float32).reshape((-1, 4))
                        file_names.append(data_path)
                if file.endswith('.pcd'):
                    V.draw_pcd(data_path)
                frames.append(points[:, :3])
                if (len(frames) > 200):
                    break
        V.draw_scenes_frames(frames, file_names, auto=True, color=True, frame_rate=cfg.frame_rate)
        if not OPEN3D_FLAG:
            mlab.show(stop=True)


def visualize_dataset():
    args, cfg = parse_config()

    index = args.index
    if args.dataset == 'waymo':
        root_data = '/media/msun/Seagate/waymo'
        WAYMO_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
        train_infos = pickle.load(open('/media/msun/Seagate/waymo/waymo_processed_data_v0_5_0_infos_train.pkl', 'rb'))
        points = []
        gt_boxes = []
        gt_label = []
        for i in range(args.frames):

            train_info = train_infos[index - i]
            data_path = os.path.join(root_data, 'waymo_processed_data_v0_5_0',
                                     train_info['point_cloud']['lidar_sequence'] + '/{:04d}.npy'.format(
                                         train_info['point_cloud']['sample_idx']))
            points.append(np.load(data_path))
            if train_info['point_cloud']['sample_idx'] == 0:
                break
            gt_boxes.append(train_info['annos']['gt_boxes_lidar'])
            gt_label_mask = np.zeros(gt_boxes[-1].shape[0], dtype=bool)
            for j, label in enumerate(train_info['annos']['name']):
                if label in WAYMO_CLASSES:
                    gt_label_mask[j] = True
                    gt_label.append(WAYMO_CLASSES.index(label) + 3 * i)
            gt_boxes[-1] = gt_boxes[-1][gt_label_mask]

        points = np.concatenate(points, axis=0)
        gt_boxes = np.concatenate(gt_boxes, axis=0)

        V.draw_scenes(
            points=points, gt_boxes=gt_boxes[:, :7], gt_labels=gt_label
        )


def visualize_file():
    args, cfg = parse_config()
    # data_file = args.data_file
    if args.data_path.endswith('.npy'):
        V.draw_scenes(
            points=np.load(args.data_path),
        )

def visualize_files():
    args,cfg = parse_config()
    files=args.data_path.split(',')
    V.draw_scenes(points=np.concatenate([np.load(file)[:,:3] for file in files]))


if __name__ == '__main__':
    args, cfg = parse_config()
    if args.func == 'visualize_dataset':
        visualize_dataset()
    elif args.func == 'visualize_file':
        visualize_file()
    elif args.func == 'visualize_files':
        visualize_files()
