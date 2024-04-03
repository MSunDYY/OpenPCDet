/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "box2map.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void box2mapLauncher(const int N,const int C,const int H,const int W,const float *boxes,const float *values,float * map);


int box2map_gpu(at::Tensor boxes_tensor, at::Tensor map_tensor, at::Tensor values_tensor)
{

    CHECK_INPUT(boxes_tensor);
    CHECK_INPUT(map_tensor);
    CHECK_INPUT(values_tensor);
    int N=boxes_tensor.size(0);
    int H=map_tensor.size(0);
    int W=map_tensor.size(1);
    int C=map_tensor.size(2);
    const float * boxes=boxes_tensor.data<float>();
    const float *values=values_tensor.data<float>();
    float * map=map_tensor.data<float>();
    box2mapLauncher(N,C,H,W,boxes,values, map);
    return 1;
}




