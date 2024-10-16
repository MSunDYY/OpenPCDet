/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 4
#define THREADS_PER_BLOCK_P 1
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
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

__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
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

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
           b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

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
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
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
#ifdef DEBUG
                printf("Cross points (%.3f, %.3f): a(%.3f, %.3f)->(%.3f, %.3f), b(%.3f, %.3f)->(%.3f, %.3f) \n",
                    cross_points[cnt - 1].x, cross_points[cnt - 1].y,
                    box_a_corners[i].x, box_a_corners[i].y, box_a_corners[i + 1].x, box_a_corners[i + 1].y,
                    box_b_corners[i].x, box_b_corners[i].y, box_b_corners[i + 1].x, box_b_corners[i + 1].y);
#endif
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
#ifdef DEBUG
                printf("b corners in a: corner_b(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
#ifdef DEBUG
                printf("a corners in b: corner_a(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
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

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}
// __global__ void mask2ind_kernel(const int N,const int P,int *points_mask_pr)
// {
//     const int idx = blockIdx.x*THREADS_PER_BLOCK_P+threadIdx.x;
//     if(idx>=N)
//     {return;}
//
//     int *points_mask_pr_cur = points_mask_pr+idx*P;
//     int num_points=0;
//     for(int i =0;i<P;i++)
//     {
//         if (points_mask_pr_cur[i]==0)
//         {
//         continue;
//         }
//         else
//         {
//         num_points+=1;
//         points_mask_pr_cur[i]=num_points;
//         }
//     }
//
// }

__global__ void points2box_kernel(const int N,const int P,const int *points_mask_pr,int *sampled_mask_pr,int*sampled_idx_pr,int *point_sampled_num_pr,const int num_sampled_per_box,const int num_sampled_per_point)
{
    const int idx = blockIdx.x*THREADS_PER_BLOCK_P+threadIdx.x;
    if(idx>=P)
    {return;}
//     if(idx%32==0)
//     {
//      printf("%d ",idx);
//      }
//     float * points_mask_pr_cur = points_mask_pr+idx;
    for(int i=0;i<N;i++)
    {
         int point_sampled_num=0;
        if((*(points_mask_pr+idx+i*P)!=0) )
        {
            if((*(points_mask_pr+idx+i*P)<=num_sampled_per_box) )
            {
            *(sampled_mask_pr+*(points_mask_pr+idx+i*P)+num_sampled_per_box*i)=1;
            *(sampled_idx_pr+*(points_mask_pr+idx+i*P)+num_sampled_per_box*i)=idx;
            point_sampled_num+=1;
            }

        }


        else
        {
        continue;
        }
    }
}



__global__ void distrituted_sample_points_kernel(const int B,const int N,const int n,const int f,const int num_points,float *voxels,bool *voxels_mask,float *srces,float *boxes)
{
    const int idx = blockIdx.x*THREADS_PER_BLOCK_P+threadIdx.x;
    if(idx>=B)
    {return;}
    const bool *voxel_mask = voxels_mask+N*idx;
    const float *box = boxes+2*idx;
    const float *src = srces+num_points*f*idx;


    for(int i=0;i<N;i++)
    {
        if (voxel_mask[i]==0)
        {continue;}
        else
        {
            continue;
        }
    }

}
__global__ void calculate_miou_kernel(float * miou,bool * point_mask,int * label,const int N,const int M)
{
    const int a_idx = blockIdx.y*THREADS_PER_BLOCK_P+threadIdx.y;
    const int b_idx = blockIdx.x*THREADS_PER_BLOCK_P+threadIdx.x;
    if(a_idx>=M || b_idx>=M || a_idx==b_idx ||  *(label+a_idx)!=*(label+b_idx)){

    return;
    }
    const bool *point_a = point_mask+a_idx*N;
    const bool *point_b = point_mask+b_idx*N;
    float Intersection=0;
    float Union=0;

    for(int i=0;i<N;i++)
    {
        Intersection += point_a[i]*point_b[i];
        Union += bool(point_a[i]+point_b[i]);

    }
    miou[a_idx*M+b_idx]=Intersection/max(static_cast<float>(Union),1.);
    miou[b_idx*M+a_idx]=Intersection/max(static_cast<float>(Union),1.);
}

__global__ void box2map_kernel(const int N ,const int C,const int H,const int W,const float *boxes,const float *values,float * map )
{
    const int idx =blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;
    if(idx>=N)
    {
    return;
    }


    const float * box = boxes+idx*7;
    const float * value =values+idx*C;
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
                if( ((A1*y_mid+B1*x_mid+C1)*(A3*y_mid+B3*x_mid+C3)<=0) && ((A2*y_mid+B2*x_mid+C2)*(A4*y_mid+B4*x_mid+C4)<=0) &&(map[(i*W+j)*C]==0))
                {

                    for (int c=0;c<C;c++)
                    {
                        map[(i*W+j)*C+c] = values[C*idx+c];
                    }
                }
            }
          }

}




void box2mapLauncher(const int N,const int C,const int H,const int W,const float *boxes,const float *values,float *map)
{
    dim3 blocks(DIVUP(N,THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    box2map_kernel<<<blocks,threads>>>(N,C,H,W,boxes,values,map);

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}


void points2boxLauncher(const int N,const int P,int *points_mask_pr,int *sampled_mask_pr,int*sampled_idx_pr,int *point_sampled_num_pr,const int num_sampled_per_box,const int num_sampled_per_point,const int num_threads)
{

//     dim3 blocks(DIBUP(N,THREADS_PER_BLOCK_P);
//     dim3 threads(THREADS_PER_BLOCK_P);
//     mask2ind_kernel<<<blocks,threads>>>(N,P,points_mask_pr);

    dim3 blocks(DIVUP(P,THREADS_PER_BLOCK_P));
    dim3 threads(THREADS_PER_BLOCK_P);
    points2box_kernel<<<blocks,threads>>>(N,P,points_mask_pr,sampled_mask_pr,sampled_idx_pr,point_sampled_num_pr,num_sampled_per_box,num_sampled_per_point);

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}

void calculate_miou_Launcher(float *miou,bool *point_mask,int *label,const int N,const int M)
{
    dim3 blocks(DIVUP(M,THREADS_PER_BLOCK_P),
                DIVUP(M,THREADS_PER_BLOCK_P));
    dim3 threads(THREADS_PER_BLOCK_P);
    calculate_miou_kernel<<<blocks,threads>>>(miou,point_mask,label,N,M);

}

void distributed_sample_points_Launcher(const int B,const int N,const int n,const int f,const int num_points,float *voxel_pr,bool *voxel_mask_pr,float *src_pr,float *boxes_pr)
{
    dim3 blocks(DIVUP(B,THREADS_PER_BLOCK_P));
    dim3 threads(THREADS_PER_BLOCK_P);
    distrituted_sample_points_kernel<<<blocks,threads>>>(B,N,n,f,num_points,voxel_pr,voxel_mask_pr,src_pr,boxes_pr);
}