from collections import namedtuple
from pcdet import device
import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):

    for key, val in batch_dict.items():
        if key in ['camera_imgs','roi_list']:
            batch_dict[key] = val.to(device)
        elif not isinstance(val, np.ndarray):
            if key=='targets_dict':
                for sub_key,sub_val in val.items():
                    batch_dict[key][sub_key]=sub_val.to(device)
            elif key=='anchors':
                batch_dict[key] = val.to(device)
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().to(device).contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().to(device)
        else:
            batch_dict[key] = torch.from_numpy(val).to(device)


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
