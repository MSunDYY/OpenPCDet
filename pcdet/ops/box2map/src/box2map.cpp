/*
3D Rotated IoU Calculation (CPU)
Written by Shaoshuai Shi
All Rights Reserved 2020.
*/

#include <stdio.h>
#include <math.h>
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

inline float min(float a, float b){
    return a > b ? b : a;
}

inline float max(float a, float b){
    return a > b ? a : b;
}

const float EPS = 1e-8;
struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

inline float box_overlap(const float *box_a, const float *box_b){
    // params: box_a (7) [x, y, z, dx, dy, dz, heading]
    // params: box_b (7) [x, y, z, dx, dy, dz, heading]

//    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3], a_angle = box_a[4];
//    float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

inline float iou_bev(const float *box_a, const float *box_b){
    // params: box_a (7) [x, y, z, dx, dy, dz, heading]
    // params: box_b (7) [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

int sample_anchor(at::Tensor miou_max_tensor,at::Tensor miou_index_tensor,at::Tensor anchors_idx_tensor,at::Tensor address1_tensor,at::Tensor address2_tensor,const float threshold)
{
    CHECK_CONTIGUOUS(miou_index_tensor);
    CHECK_CONTIGUOUS(anchors_idx_tensor);


    float * miou_max = miou_max_tensor.data<float>();
    int * miou_index = miou_index_tensor.data<int>();

    int * anchors_idx = anchors_idx_tensor.data<int>();
    int *address1 = address1_tensor.data<int>();
    int *address2 = address2_tensor.data<int>();
    int N = miou_max_tensor.size(0);

    int location=0;
    for(int i=0;i<N;i++)
    {
        if (miou_index[i]<i && miou_max[i]>threshold )
        {
            address1[i]=address1[miou_index[i]];

            anchors_idx[address1[i]*N+address2[address1[i]]]=i;
            address2[address1[i]]+=1;

        }
        else
        {
            anchors_idx[location*N]=i;
            address1[i] = location;
            address2[address1[i]]+=1;
            location++;
        }

    }
    return location;
}


int box2map(at::Tensor boxes_tensor, at::Tensor map_tensor, at::Tensor values_tensor){
    // params boxes_a_tensor: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b_tensor: (H,W,C)
    // params ans_iou_tensor: (N,C)
    //the func it to ouptput the values of boxes to corrispoding location of map
    CHECK_CONTIGUOUS(boxes_tensor);
    CHECK_CONTIGUOUS(map_tensor);
    CHECK_CONTIGUOUS(values_tensor);

    int N = boxes_tensor.size(0);
    int H = map_tensor.size(0);
    int W = map_tensor.size(1);
    int C = map_tensor.size(2);

    const float *boxes = boxes_tensor.data<float>();
    float *map = map_tensor.data<float>();
    const float *values = values_tensor.data<float>();


    for (int n = 0; n < N; n++){
        const float *box = boxes+n*7;


        float x = box[0];
        float y = box[1];
        float l = box[3];
        float w = box[4];
        float theta = box[6];
        float y_right = y+(abs(l*cos(theta)) + abs(w*sin(theta))) /2;
        float x_top = x+(abs(l*sin(theta)) + abs(w*cos(theta))) /2;
        float y_left = y-(abs(l*cos(theta)) + abs(w*sin(theta))) /2;
        float x_bottom = x-(abs(l*sin(theta)) + abs(w*cos(theta))) /2;
        float x_right,x_left,y_top,y_bottom;

        if(theta<=M_PI/2)
        {
            x_right = x+(l*sin(theta)-w*cos(theta))/2;
            x_left  = x-(l*sin(theta)-w*cos(theta))/2;
            y_top = y+(l*cos(theta)-w*sin(theta))/2;
            y_bottom = y-(l*cos(theta)-w*sin(theta))/2;
        }
        else
        {
            x_right = x-(l*sin(theta)-w*cos(theta))/2;
            x_left = x+(l*cos(theta)+w*cos(theta))/2;
            y_top = y+(l*cos(theta)+w*sin(theta))/2;
            y_bottom = y-(l*cos(theta)+w*sin(theta))/2;
        }
        float A1 = x_top-x_left;
        float B1 = y_left-y_top;
        float C1 = y_top*x_left - y_left*x_top;

        float A2 = x_right-x_top;
        float B2 = y_top-y_right;
        float C2 = y_right*x_right-y_top*x_right;

        float A3 = x_right-x_bottom;
        float B3 = y_bottom-y_right;
        float C3 = y_right*x_bottom -y_bottom*x_right;

        float A4 = x_left - x_bottom;
        float B4 = y_bottom-y_left;
        float C4 = y_left*x_bottom-y_bottom*x_left;


        for (int i=max(static_cast<int>(x_bottom),0);i<min(static_cast<int>(x_top),H);i++)
        {
            for(int j=max(static_cast<int>(y_left),0);j<min(static_cast<int>(y_right),W);j++)
            {

                float y_mid = j+0.5;
                float x_mid =i+0.5;
                if( ((A1*y_mid+B1*x_mid+C1)*(A3*y_mid+B3*x_mid+C3)<=0) && ((A2*y_mid+B2*x_mid+C2)*(A4*y_mid+B4*x_mid+C4)<=0))
                {

                    for (int c=0;c<C;c++)
                    {
                        map[(i*W+j)*C+c] = values[C*n+c];
                    }
                }
            }
          }

    }
    return 1;
}



int boxes_aligned_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor){
    // params boxes_a_tensor: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b_tensor: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_iou_tensor: (N, 1)

    CHECK_CONTIGUOUS(boxes_a_tensor);
    CHECK_CONTIGUOUS(boxes_b_tensor);

    int num_boxes = boxes_a_tensor.size(0);
    int num_boxes_b = boxes_b_tensor.size(0);
    assert(num_boxes == num_boxes_b);
    const float *boxes_a = boxes_a_tensor.data<float>();
    const float *boxes_b = boxes_b_tensor.data<float>();
    float *ans_iou = ans_iou_tensor.data<float>();

    for (int i = 0; i < num_boxes; i++){
        ans_iou[i] = iou_bev(boxes_a + i * 7, boxes_b + i * 7);
    }
    return 1;
}

int points2box(at::Tensor points_mask,at::Tensor sampled_mask,at::Tensor sampled_idx,at::Tensor point_sampled_num,const int num_sampled_per_box,const int num_sampled_per_point)
{
    int N = points_mask.size(0);
    int P = points_mask.size(1);
    const int *points_mask_pr = points_mask.data<int>();
    int *sampled_mask_pr = sampled_mask.data<int>();
    int *sampled_idx_pr = sampled_idx.data<int>();
    int *point_sampled_num_pr = point_sampled_num.data<int>();


    for(int idx=0;idx<P;idx++)
    {
        for(int i=0;i<N;i++)
        {
            int point_sampled_num=0;
            if((*(points_mask_pr+idx+i*P)==1) &&(point_sampled_num_pr[i]<num_sampled_per_box))
            {
                *(sampled_mask_pr+point_sampled_num_pr[i]+num_sampled_per_box*i)=1;
                *(sampled_idx_pr+point_sampled_num_pr[i]+num_sampled_per_box*i)=idx;
                point_sampled_num_pr[i]+=1;
                point_sampled_num+=1;
                if (point_sampled_num==num_sampled_per_point)
                {
                break;
                }
            }
            else
            {
            continue;
            }
        }
    }
    return 1;
}
