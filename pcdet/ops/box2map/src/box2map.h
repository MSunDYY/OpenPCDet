#ifndef BOX2MAP_H
#define BOX2MAP_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int box2map(at::Tensor boxes_tensor, at::Tensor map_tensor, at::Tensor values_tensor);
int boxes_aligned_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor);
int box2map_gpu(at::Tensor boxes_tensor,at::Tensor map_tensor,at::Tensor values_tensor);
int points2box_gpu(at::Tensor points_mask,at::Tensor sampled_mask,at::Tensor sampled_idx,at::Tensor point_sampled_num,const int num_sampled_per_box,const int num_sampled_per_point,const int num_threads);
int points2box(at::Tensor points_mask,at::Tensor sampled_mask,at::Tensor sampled_idx,at::Tensor point_sampled_num,const int num_sampled_per_box,const int num_sampled_per_point);
int distributed_sample_points(at::Tensor voxels,at::Tensor voxel_mask,at::Tensor src,at::Tensor boxes,const int num_points);
int sample_anchor(at::Tensor miou_max_tensor,at::Tensor miou_index_tensor,at::Tensor anchors_idx_tensor,at::Tensor address1_tensor,at::Tensor address2_tensor,const float threshold);
int calculate_miou_gpu(at::Tensor miou_tensor,at::Tensor point_mask_tensor,at::Tensor label_tensor);

#endif
