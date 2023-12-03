import torch.nn as nn
import torch
from ..detectors.detector3d_template import Detector3DTemplate
from pcdet import device

class Sampler(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.linear1 = nn.Linear(128,64)
        self.linear2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()




    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_function = torch.nn.BCEWithLogitsLoss()
        label = batch_dict['key_points_label']
        loss = loss_function(label,batch_dict['predict_class'].squeeze())
        tb_dict = {'cls_loss':loss}
        return loss, tb_dict,disp_dict

    def forward(self, batch_dict):

        for module in self.module_list:
            batch_dict = module(batch_dict)
        batch_dict['predict_class'] =self.sigmoid(self.linear2(self.linear1(batch_dict['point_features']))).squeeze()

        if self.training:
            loss , tb_dict,disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {'loss':loss}
            return ret_dict , tb_dict,disp_dict
        else:
            return batch_dict
