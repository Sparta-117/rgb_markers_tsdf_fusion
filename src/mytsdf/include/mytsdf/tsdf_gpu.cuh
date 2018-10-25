#ifndef TSDF_GPU_CUH
#define TSDF_GPU_CUH

//#include "common_include.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

// for Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vector>
#include <opencv2/opencv.hpp>

//using namespace std;
//using namespace cv;

//class TSDFGpu
//{
//public:
    //typedef std::shared_ptr<TSDFGpu> Ptr;
    __global__ void Integrate(float * cam_K, float * cam2base, float * depth_im, unsigned char * rgb_im,
                   int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                   float * voxel_grid_TSDF, float * voxel_grid_weight,
                   unsigned char * voxel_grid_rgb, float * voxel_grid_rgb_weight, float * voxel_grid_rgb_diff);
    // Compute surface points from TSDF voxel grid and save points to point cloud file
    void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                         float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                         float * voxel_grid_TSDF, float * voxel_grid_weight, unsigned char * voxel_grid_rgb,
                                         float tsdf_thresh, float weight_thresh);
    // Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
    // Return the matrix as a float vector of the matrix in row-major order
    std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N);
    // Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
    // The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
    void ReadDepth(std::string filename, int H, int W, float * depth, int depth_scale);
    void ReadDepth(cv::Mat depth_mat, int H, int W, float * depth, int depth_scale);
    void ReadRGB(cv::Mat rgb_mat, int H, int W, unsigned char * rgb);
    // 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
    void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);
    // 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
    bool invert_matrix(const float m[16], float invOut[16]);
    void FatalError(const int lineNumber = 0);
    void checkCUDA(const int lineNumber, cudaError_t status);
    std::vector<float> Matrix2Array(cv::Matx33f input, int M, int N);
    std::vector<float> Matrix2Array(Eigen::Matrix4f input, int M, int N);
    void RunKernal(float * cam_K, float * cam2base, float * depth_im, unsigned char * rgb_im,
                    int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                    float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                    float * voxel_grid_TSDF, float * voxel_grid_weight,
                    unsigned char * voxel_grid_rgb, float * voxel_grid_rgb_weight, float * voxel_grid_rgb_diff);
//};

#endif // TSDF_GPU_CUH
