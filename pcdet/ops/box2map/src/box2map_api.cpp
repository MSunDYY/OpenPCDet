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
	m.def("points2box_gpu",&points2box_gpu,"selete points to box");
    m.def("points2box",&points2box,"cpu");
    m.def("distributed_smaple_points",&distributed_sample_points,"distributed_sample_points");
}
