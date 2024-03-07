#ifndef BOX2MAP_H
#define BOX2MAP_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int box2map(at::Tensor boxes_tensor, at::Tensor map_tensor, at::Tensor values_tensor);
int boxes_aligned_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor);
int box2map_gpu(at::Tensor boxes_tensor,at::Tensor map_tensor,at::Tensor values_tensor);
#endif
