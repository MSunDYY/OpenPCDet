"""
References:
PointPWC-Net: https://github.com/DylanWusee/PointPWC
FLOT: https://github.com/valeoai/FLOT
FlowStep3D: https://github.com/yairkit/flowstep3d
"""

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import ot
from utils.graph import Graph
from utils.gconv import SetConv
from utils.modules import learn_SLIC_calc_mutual
from lib.pointops.functions import pointops
from ..detectors.detector3d_template import Detector3DTemplate

from utils.pointconv_util import UpsampleFlow, FlowEmbedding, PointWarping, index_points_gather, PointWarping_feat


class GRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(GRU, self).__init__()
        in_ch = hidden_dim + input_dim
        self.convz = SetConv(in_ch, hidden_dim)
        self.convr = SetConv(in_ch, hidden_dim)
        self.convq = SetConv(in_ch, hidden_dim)

    def forward(self, h, x, c, graph):
        hx = torch.cat([h, x], dim=2)
        z = torch.sigmoid(self.convz(hx, graph))
        r = torch.sigmoid(self.convr(hx, graph))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=2), graph))
        h = (1 - z) * h + z * q
        return h


class FlowNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()


    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.8))
        key_pillars_pred = batch_dict['key_pillars_pred']
        key_pillars_label = batch_dict['key_pillars_label']
        random_index = torch.randint(0, key_pillars_label.shape[0], (int(2 * key_pillars_label.sum().item()),))
        train_mask = torch.zeros_like(key_pillars_label).bool()
        train_mask[random_index] = True
        train_mask += key_pillars_label.bool()

        key_pillars_pred = key_pillars_pred[train_mask]
        key_pillars_label = key_pillars_label[train_mask]

        loss = loss_function(key_pillars_pred, key_pillars_label)

        if loss.item() < 0:
            pass
        tb_dict = {'cls_loss': loss}
        return loss, tb_dict, disp_dict

    def forward(self, batch_dict):

        for module in self.module_list:
            if getattr(module,'is_train',True) is False:
                with torch.no_grad():
                    batch_dict = module(batch_dict)
            else:
                batch_dict = module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:

            batch_dict['key_pillars_pred'] = self.sigmoid(batch_dict['key_pillars_pred'])
            return batch_dict
