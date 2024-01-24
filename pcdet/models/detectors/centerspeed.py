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
            dataset.data_augmentor.data_augmentor_queue.pop(0)
        else:
            dataset.dataset_cfg['SEQUENCE_CONFIG'].ENABLED = False
            dataset.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES-=1
            dataset.CONCAT = True
            dataset.dataset_cfg.DATA_PROCESSOR[2].CONCAT=True
    def forward(self, batch_dict):


        if self.training:
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
                    pred_dict['coordinate_all'] = batch_dict['coordinate_all']
                    pred_dict['speed_all'] = batch_dict['speed_all']
                    pred_dict['speed_compressed'] = batch_dict['speed_map_pred']

                useless_para = ['multi_scale_3d_features','multi_scale_3d_strides','spatial_features','spatial_features_stride']
                for para in useless_para:
                    batch_dict.pop(para)

        else:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

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
