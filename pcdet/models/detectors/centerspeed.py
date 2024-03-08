import spconv.pytorch
import torch

from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes
from pcdet.models.preprocess.speed_estimate import SpeedEstimater

class CenterSpeed(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.speed_est = SpeedEstimater()
        self.train_box = model_cfg.DENSE_HEAD.TRAIN_BOX


        if not model_cfg.DENSE_HEAD.TRAIN_BOX:
            if dataset.training:
                dataset.data_augmentor.data_augmentor_queue.pop(0)
        else:
            if dataset.training:
                dataset.data_augmentor.data_augmentor_queue.pop(1)
            dataset.dataset_cfg['SEQUENCE_CONFIG'].ENABLED = False
            if dataset.training:
                dataset.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES-=1
            dataset.CONCAT = True
            dataset.dataset_cfg.DATA_PROCESSOR[2].CONCAT=True
            dataset.dataset_cfg.DATA_PROCESSOR[3].CONCAT=True
            dataset.data_processor.data_processor_queue.pop(-1)
    def forward(self, batch_dict):



        if self.train_box:
            for cur_module in self.module_list[1:]:
                batch_dict = cur_module(batch_dict)
        else:
            batch_dict  = self.module_list[0](batch_dict)
            with torch.no_grad():
                for cur_module in self.module_list[1:]:
                    batch_dict = cur_module(batch_dict)

            for pred_dict in self.dense_head.forward_ret_dict['pred_dicts']:
                pred_dict['is_moving'] = batch_dict['is_moving']
                pred_dict['coordinate_all'] = batch_dict['pillar_coords']
                pred_dict['speed_1st'] = batch_dict['speed_1st']
                pred_dict['speed_compressed'] = batch_dict['speed_map_pred']

            useless_para = ['multi_scale_3d_features','multi_scale_3d_strides','spatial_features','spatial_features_stride','voxel_features']
            for para in useless_para:
                del batch_dict[para]
        torch.cuda.empty_cache()


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
        is_moving_mask = batch_dict['is_moving']>0.5
        speed_pred = batch_dict['speed_1st']
        pillar_coords = batch_dict['pillar_coords']
        if not self.train_box:
            for index in range(batch_size):
                pred_boxes = final_pred_dict[index]['pred_boxes']

                recall_dict = self.generate_recall_record(
                    box_preds=pred_boxes,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    visualization=False
                )
        else:
            speed_sp_tensor = spconv.pytorch.SparseConvTensor(
                features=speed_pred,
                indices= pillar_coords,
                spatial_shape=[4,1504,1504],
                batch_size=self.dense_head.B

            )

            speed_dense_tensor = speed_sp_tensor.dense()

            for index in range(self.dense_head.B):
                pred_boxes = final_pred_dict[index*self.F]['pred_boxes']
                for f in range(1,self.dense_head.F):
                    pred_boxes_pre = final_pred_dict[index*self.dense_head.F+f]
                    coord_mask = (pillar_coords[:,0]==index) * (pillar_coords[:,1]==f)
                    pred_boxes_pre_coor = (pred_boxes_pre[:,:2]-self.preprocess.point_cloud_range[:2])//torch.tensor(self.preprocess.voxel_size[:2]).to(device)
                    pred_boxes_pre[:,:2]+= speed_dense_tensor[index,F,pred_boxes_pre_coor[:,1],pred_boxes_pre_coor[:,0]]


        return final_pred_dict, recall_dict
