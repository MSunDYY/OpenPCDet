from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes
import torch
from pcdet import device

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()


    def forward(self, batch_dict):
        from spconv.utils import Point2VoxelGPU3d as VoxelGenerator
        import cumm.tensorview as tv
        if self.dataset.dataset_cfg.DATA_PROCESSOR[-1].NAME=='transform_points_to_voxels':
            if self.dataset.dataset_cfg.DATA_PROCESSOR[-1].get('GPU',False):
                import cumm.tensorview as tv
                voxel_generator = VoxelGenerator(
                    vsize_xyz=self.dataset.data_processor.voxel_size,
                    coors_range_xyz = self.dataset.data_processor.point_cloud_range,
                    num_point_features=self.dataset.data_processor.num_point_features,
                    max_num_points_per_voxel=self.dataset.dataset_cfg.DATA_PROCESSOR[-1].MAX_POINTS_PER_VOXEL,
                    max_num_voxels=self.dataset.dataset_cfg.DATA_PROCESSOR[-1].MAX_NUMBER_OF_VOXELS[self.mode.lower()]
                )
                voxel_list=[]
                coordinates_list = []
                num_voxel_points_list = []
                points = batch_dict['points']
                for b in range(batch_dict['batch_size']):
                    points_b = tv.from_numpy(points[points[:,0]==b][:,1:].to('cpu').numpy()).cuda()
                    voxel_output = voxel_generator.point_to_voxel_hash(points_b)
                    voxels,coordinates,num_points = voxel_output
                    voxel_list.append(torch.from_numpy(voxels.cpu().numpy()).to(device))
                    coordinates = torch.from_numpy(coordinates.cpu().numpy()).to(device)
                    coordinates_list.append(torch.cat([torch.ones([coordinates.shape[0],1],dtype=coordinates.dtype).to(device)*b,coordinates],dim=1))
                    num_voxel_points_list.append(torch.from_numpy(num_points.cpu().numpy()).to(device))
                batch_dict['voxels'] = torch.concat(voxel_list,dim=0)
                batch_dict['voxel_coords'] = torch.concat(coordinates_list,dim=0)
                batch_dict['voxel_num_points'] = torch.concat(num_voxel_points_list,dim=0)

                # if not batch_dict['use_lead_xyz']:
                #     voxels = voxels[...,3:]

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        import time
        time.sleep(2)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                visualization=False
            )


        return final_pred_dict, recall_dict
