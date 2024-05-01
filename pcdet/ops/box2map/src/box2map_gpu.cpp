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
void points2boxLauncher(const int N,const int P,int *points_mask_pr,int *sampled_mask_pr,int*sampled_idx_pr,int *point_sampled_num_pr,const int num_sampled_per_box,const int num_sampled_per_point,const int num_threads);
void distributed_sample_points_Launcher(const int B,const int N,const int n,const int f,const int num_points,float *voxel_pr,bool *voxel_mask_pr,float *src_pr,float *boxes);


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

int points2box_gpu(at::Tensor points_mask,at::Tensor sampled_mask,at::Tensor sampled_idx,at::Tensor point_sampled_num,const int num_sampled_per_box,const int num_sampled_per_point,const int num_threads)
{
    CHECK_INPUT(points_mask);
    CHECK_INPUT(sampled_mask);
    CHECK_INPUT(sampled_idx);
    CHECK_INPUT(point_sampled_num);
    int N = points_mask.size(0);
    int P = points_mask.size(1);
    CHECK_INPUT(points_mask);
    CHECK_INPUT(sampled_idx);
    CHECK_INPUT(sampled_mask);
    int *points_mask_pr = points_mask.data<int>();
    int *sampled_mask_pr = sampled_mask.data<int>();
    int *sampled_idx_pr = sampled_idx.data<int>();
    int *point_sampled_num_pr = point_sampled_num.data<int>();
    points2boxLauncher(N,P,points_mask_pr,sampled_mask_pr,sampled_idx_pr,point_sampled_num_pr,num_sampled_per_box,num_sampled_per_point,num_threads);
    return 1;
}

int distributed_sample_points(at::Tensor voxels,at::Tensor voxel_mask,at::Tensor src,at::Tensor boxes,const int num_points);
{
    CHECK_INPUT(voxels);
    CHECK_INPUT(voxel_mask);
    CHECK_INPUT(src);
    int B=src.size(0);
    int N=voxel_mask.size(0);
    int n=voxels.size(1);
    int f=voxels.size(2);
    float *voxel_pr = voxels.data<float>();
    bool *voxel_mask_pr = voxel_mask.data<bool>();
    float *src_pr = src.data<float>();
    float *boxes_pr = boxes.data<float>();
    distributed_sample_points_Launcher(B,N,n,f,num_points,voxel_pr,voxel_mask_pr,src_pr,boxes_pr);
}


