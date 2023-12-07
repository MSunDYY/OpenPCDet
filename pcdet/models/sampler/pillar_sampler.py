import torch.nn as nn
import torch
from ..detectors.detector3d_template import Detector3DTemplate
from pcdet import device
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper


class PillarSampler(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.linear1 = nn.Linear(128,64)
        self.linear2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()




    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=0.8)
        key_pillars_pred = batch_dict['key_pillars_pred']
        key_pillars_label = batch_dict['key_pillars_label']
        loss = loss_function(key_pillars_pred,key_pillars_label)


        if loss.item()<0:
            pass
        tb_dict = {'cls_loss':loss}
        return loss, tb_dict,disp_dict

    def forward(self, batch_dict):

        for module in self.module_list:
            batch_dict = module(batch_dict)


        if self.training:
            loss , tb_dict,disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {'loss':loss}
            return ret_dict , tb_dict,disp_dict
        else:
            return batch_dict
