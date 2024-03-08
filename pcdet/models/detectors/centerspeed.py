import spconv.pytorch
import torch
from pcdet import device
from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes
from pcdet.models.preprocess.speed_estimate import SpeedEstimater


class CenterSpeed(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.speed_est = SpeedEstimater()
        self.train_box = model_cfg.DENSE_HEAD.TRAIN_BOX
        self.pillar_size = self.dataset.dataset_cfg.DATA_PROCESSOR[-1].PILLAR_SIZE
        self.pillar_spatial_shape = [round((self.dataset.point_cloud_range[3]-self.dataset.point_cloud_range[0])/self.pillar_size[0]),
                                     round((self.dataset.point_cloud_range[4]-self.dataset.point_cloud_range[1])/self.pillar_size[1])]
        if not model_cfg.DENSE_HEAD.TRAIN_BOX:
            if dataset.training:
                dataset.data_augmentor.data_augmentor_queue.pop(0)
        else:
            if dataset.training:
                dataset.data_augmentor.data_augmentor_queue.pop(1)
            dataset.dataset_cfg['SEQUENCE_CONFIG'].ENABLED = False
            if dataset.training:
                dataset.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES -= 1
            dataset.CONCAT = True
            dataset.dataset_cfg.DATA_PROCESSOR[2].CONCAT = True
            dataset.dataset_cfg.DATA_PROCESSOR[3].CONCAT = True
            dataset.data_processor.data_processor_queue.pop(-1)

    def forward(self, batch_dict):

        if self.train_box:
            for cur_module in self.module_list[1:]:
                batch_dict = cur_module(batch_dict)
        else:


            batch_dict = self.module_list[0](batch_dict)
            with torch.no_grad():
                for cur_module in self.module_list[1:]:
                    batch_dict = cur_module(batch_dict)

            for pred_dict in self.dense_head.forward_ret_dict['pred_dicts']:
                pred_dict['is_moving'] = batch_dict['is_moving']
                pred_dict['coordinate_all'] = batch_dict['pillar_coords']
                pred_dict['speed_1st'] = batch_dict['speed_1st']
                pred_dict['speed_compressed'] = batch_dict['speed_map_pred']

            useless_para = ['multi_scale_3d_features', 'multi_scale_3d_strides', 'spatial_features',
                            'spatial_features_stride', 'voxel_features']
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


        if self.train_box:
            for index in range(batch_size):
                pred_boxes = final_pred_dict[index]['pred_boxes']

                recall_dict = self.generate_recall_record(
                    box_preds=pred_boxes,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    visualization=False
                )
        else:
            B = self.dense_head.B
            F = self.dense_head.F
            is_moving_mask = batch_dict['is_moving'] > 0.5
            speed_pred = batch_dict['speed_1st']
            pillar_coords = batch_dict['pillar_coords']
            gt_boxes = batch_dict['gt_boxes']
            gt_boxes = [gt_box[gt_box[:, -2] == 0] for gt_box in gt_boxes]
            gt_boxes_num = [gt_box.shape[0] for gt_box in gt_boxes]
            gt_boxes_temp = torch.zeros((B, max(gt_boxes_num), 10)).to(device)
            batch_dict['gt_boxes'] = gt_boxes_temp

            for b in range(B):
                gt_boxes_temp[b,:gt_boxes_num[b],:-1] = gt_boxes[b][:,:-2]
                gt_boxes_temp[b,:gt_boxes_num[b],-1] = gt_boxes[b][:,-1]
            speed_sp_tensor = spconv.pytorch.SparseConvTensor(
                features=speed_pred,
                indices=pillar_coords.int(),
                spatial_shape=[F]+self.pillar_spatial_shape,
                batch_size=B
            )

            speed_dense_tensor = speed_sp_tensor.dense()
            speed_dense_tensor = speed_dense_tensor.permute(0, 2, 3, 4, 1)
            cor_final_pred_dict = []
            for index in range(B):
                pred_boxes = final_pred_dict[index * F]['pred_boxes']
                pred_dict_temp = {'pred_boxes':[],'pred_scores':[],'pred_labels':[]}
                for f in range(1, F):
                    pred_boxes_pre = final_pred_dict[index * F + f]['pred_boxes']
                    coord_mask = (pillar_coords[:, 0] == index) * (pillar_coords[:, 1] == f)
                    pred_boxes_pre_coor = ((pred_boxes_pre[:, :2] - torch.from_numpy(
                        self.preprocess.point_cloud_range[:2]).to(device)) // torch.tensor(
                        self.pillar_size[:2]).to(device))
                    pred_boxes_pre[:, :2] += speed_dense_tensor[
                        index, f][pred_boxes_pre_coor[:, 0].long(), pred_boxes_pre_coor[:, 1].long()]*0.1*f
                    pred_dict_temp['pred_boxes'].append(pred_boxes_pre)
                    pred_dict_temp['pred_labels'].append(final_pred_dict[index*F+f]['pred_labels'])
                    pred_dict_temp['pred_scores'].append(final_pred_dict[index*F+f]['pred_scores'])
                pred_dict_temp['pred_boxes'] = torch.concat(pred_dict_temp['pred_boxes'],dim=0)
                pred_dict_temp['pred_scores'] = torch.concat(pred_dict_temp['pred_scores'],dim=0)
                pred_dict_temp['pred_labels'] = torch.concat(pred_dict_temp['pred_labels'],dim=0)
                cor_final_pred_dict.append(pred_dict_temp)
                recall_dict = self.generate_recall_record(
                    box_preds=pred_dict_temp['pred_boxes'],
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    visualization=False
                )
            final_pred_dict = cor_final_pred_dict
        return final_pred_dict, recall_dict
