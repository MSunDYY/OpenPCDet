#ifndef BOX2MAP_H
#define BOX2MAP_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int sel_trac_boxes(at::Tensor boxes, at::Tensor inds);

#endif
