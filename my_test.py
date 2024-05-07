import os

import numpy as np
import open3d as o3d

import shutil
source_path = '../data/waymo/waymo_processed_data_v0_5_0_full'
dist_path = '../data/waymo/waymo_processed_data_v0_5_0_full_val'
val_file = '../data/waymo/ImageSets/val.txt'

dir_list = os.listdir(source_path)
with open(val_file, 'r') as f:
    for line in f:
        # 处理每一行的内容，例如打印出来
        line = line[:-10]
        if line in dir_list:
            shutil.move(os.path.join(souce_path,line),shutil.move(os.path.join(dist_path,line)))


