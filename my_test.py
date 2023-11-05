import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d


file_name = 'data/waymo/waymo_processed_data_v0_5_0/segment-272435602399417322_2884_130_2904_130_with_camera_labels/0000.npy'
data = np.load(file_name)
data = data[:, :3]
point_cloud = o3d.geometry.PointCloud()
point_cloud.points=o3d.utility.Vector3dVector(data)
o3d.visualization.draw_geometries([point_cloud])

