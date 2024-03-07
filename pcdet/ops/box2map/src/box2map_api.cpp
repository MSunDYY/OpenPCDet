#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "box2map.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


	m.def("boxes_aligned_iou_bev_cpu", &boxes_aligned_iou_bev_cpu, "aligned oriented boxes iou");
	m.def("box2map", &box2map, "boxes information copied to map");
	m.def("box2map_gpu",&box2map_gpu,"boxes information copied to map");
}
