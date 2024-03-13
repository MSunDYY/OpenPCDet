import spconv.pytorch
import torch
from pcdet import device
from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes
from pcdet.models.preprocess.speed_estimate import SpeedEstimater
from pcdet.ops.box2map import box2map


class CenterSpeed(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.speed_est = SpeedEstimater()
        self.train_box = model_cfg.DENSE_HEAD.TRAIN_BOX
        self.pillar_size = self.dataset.dataset_cfg.DATA_PROCESSOR[-1].PILLAR_SIZE
        self.pillar_spatial_shape = [
            round((self.dataset.point_cloud_range[3] - self.dataset.point_cloud_range[0]) / self.pillar_size[0]),
            round((self.dataset.point_cloud_range[4] - self.dataset.point_cloud_range[1]) / self.pillar_size[1])]
        memory_max = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        if memory_max > 5000:
            dataset.dataset_cfg.DATA_PROCESSOR[-1].MAX_NUMBER_OF_PILLARS['train'] *= 1000
            dataset.dataset_cfg.DATA_PROCESSOR[-1].MAX_NUMBER_OF_PILLARS['test'] *= 1000

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
                pred_dict['is_moving_pred'] = batch_dict['is_moving_pred']
                pred_dict['coordinate_all'] = batch_dict['pillar_coords']
                pred_dict['speed_1st'] = batch_dict['speed_1st']
                pred_dict['speed_compressed_pred'] = batch_dict['speed_map_pred']

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
    def generate_speed_eval(self,speed_pillar_gt,speed_map_gt,is_moving_mask,speed_pred,final_pred_dict):
        is_moving_mask_gt = torch.pow(speed_pillar_gt,2).sum(dim=-1)>0.25
         # = is_moving_mask==is_moving_mask_gt
        speed_res = (speed_pillar_gt[is_moving_mask] - speed_pred[is_moving_mask])**2
        speed_res = torch.sqrt(torch.sum(speed_res)/speed_res.shape[0])

        speed_map_compressed_gt = speed_map_gt.reshape()

        print('speed_moving_cls:  {} / {}'.format((is_moving_mask[is_moving_mask_gt]).sum(),is_moving_mask_gt.sum()))
        print('mean_speed_res:  {:.4f}'.format(speed_res))

        return final_pred_dict
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
            is_moving_mask = batch_dict['is_moving']> 0.5
            speed_pred = batch_dict['speed_1st']
            pillar_coords = batch_dict['pillar_coords']
            gt_boxes_all = batch_dict['gt_boxes']
            gt_boxes = batch_dict['gt_boxes']
            gt_boxes = [gt_box[gt_box[:, -2] == 0] for gt_box in gt_boxes]
            gt_boxes_num = [gt_box.shape[0] for gt_box in gt_boxes]
            gt_boxes_temp = torch.zeros((B, max(gt_boxes_num), 10)).to(device)
            batch_dict['gt_boxes'] = gt_boxes_temp
            for b in range(B):
                gt_boxes_temp[b, :gt_boxes_num[b], :-1] = gt_boxes[b][:, :-2]
                gt_boxes_temp[b, :gt_boxes_num[b], -1] = gt_boxes[b][:, -1]
            speed_sp_tensor = spconv.pytorch.SparseConvTensor(
                features=speed_pred,
                indices=pillar_coords.int(),
                spatial_shape=[F] + self.pillar_spatial_shape,
                batch_size=B
            )

            speed_dense_tensor = speed_sp_tensor.dense()
            speed_dense_tensor = speed_dense_tensor.permute(0, 2, 3, 4, 1)
            cor_final_pred_dict = []
            speed_map_all = []
            for index in range(B):
                pred_boxes = final_pred_dict[index * F]['pred_boxes']
                pred_dict_temp = {'pred_boxes': [], 'pred_scores': [], 'pred_labels': []}
                gt_box_batch = gt_boxes_all[index]
                speed_map_batch = []

                speed_map_gt = torch.zeros(self.pillar_spatial_shape + [2]).to(device)

                gt_box = gt_box_batch[gt_box_batch[:,-2]==0]
                gt_box[:,:2] = (gt_box[:, :2] - torch.from_numpy(self.dataset.point_cloud_range[:2]).to(device)) / torch.tensor(
                        self.pillar_size[:2])[None, :].to(device)
                gt_box[:, 3:5] = (gt_box[:, 3:5]) / torch.tensor(self.pillar_size[:2])[None, :].to(device)
                gt_box[:, 6][gt_box[:, 6] < 0] += torch.pi
                box2map.box2map_gpu(gt_box[:, :7].contiguous(), speed_map_gt, gt_box[:, 7:9].contiguous())
                speed_map_batch.append(speed_map_gt[None, :])

                for f in range(1, F):
                    pred_boxes_pre = final_pred_dict[index * F + f]['pred_boxes']
                    coord_mask = (pillar_coords[:, 0] == index) * (pillar_coords[:, 1] == f)
                    pred_boxes_pre_coor = ((pred_boxes_pre[:, :2] - torch.from_numpy(
                        self.preprocess.point_cloud_range[:2]).to(device)) // torch.tensor(
                        self.pillar_size[:2]).to(device))
                    pred_boxes_pre[:, :2] += speed_dense_tensor[
                                                 index, f][pred_boxes_pre_coor[:, 0].long(), pred_boxes_pre_coor[:,
                                                                                             1].long()] * 0.1 * f
                    pred_dict_temp['pred_boxes'].append(pred_boxes_pre)
                    pred_dict_temp['pred_labels'].append(final_pred_dict[index * F + f]['pred_labels'])
                    pred_dict_temp['pred_scores'].append(final_pred_dict[index * F + f]['pred_scores'])
                    speed_map_gt = torch.zeros(self.pillar_spatial_shape + [2]).to(device)
                    speed_map_pred = batch_dict['speed_map_pred']
                    gt_box = gt_box_batch[gt_box_batch[:, -2] == f]
                    gt_box[:, :2] = (gt_box[:, :2] - torch.from_numpy(self.dataset.point_cloud_range[:2]).to(device)) / torch.tensor(
                        self.pillar_size[:2])[None, :].to(device)
                    gt_box[:,3:5] = (gt_box[:,3:5]) / torch.tensor(self.pillar_size[:2])[None,:].to(device)
                    gt_box[:,6][gt_box[:,6]<0]+=torch.pi
                    box2map.box2map_gpu(gt_box[:, :7].contiguous(), speed_map_gt, gt_box[:, 7:9].contiguous())
                    speed_map_batch.append(speed_map_gt[None,:])



                speed_map_batch = torch.concat(speed_map_batch,dim=0)
                speed_map_all.append(speed_map_batch[None,:])
                pred_dict_temp['pred_boxes'] = torch.concat(pred_dict_temp['pred_boxes'], dim=0)
                pred_dict_temp['pred_scores'] = torch.concat(pred_dict_temp['pred_scores'], dim=0)
                pred_dict_temp['pred_labels'] = torch.concat(pred_dict_temp['pred_labels'], dim=0)
                cor_final_pred_dict.append(pred_dict_temp)
                recall_dict = self.generate_recall_record(
                    box_preds=pred_dict_temp['pred_boxes'],
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    visualization=False
                )
            speed_map_all = torch.concat(speed_map_all,dim=0)

            speed_pillar_gt = speed_map_all[pillar_coords[:,0],pillar_coords[:,1],pillar_coords[:,2],pillar_coords[:,3]]

            final_pred_dict = cor_final_pred_dict

            speed_map_compressed = speed_map_all[...,-1]**2+speed_map_all[...,-2]**2
            speed_map_compressed_inds = (speed_map_compressed>0).sum(dim=1)

            speed_map_compressed = torch.sum(speed_map_compressed,dim=1)

            final_pred_dict = self.generate_speed_eval(speed_pillar_gt,speed_map_all,is_moving_mask,speed_pred,final_pred_dict)


        return final_pred_dict, recall_dict
