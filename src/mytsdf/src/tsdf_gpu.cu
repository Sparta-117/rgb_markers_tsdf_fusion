#include "tsdf_gpu.cuh"

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * cam2world, float * depth_im, unsigned char * rgb_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight,
               unsigned char * voxel_grid_rgb, float * voxel_grid_rgb_weight, float * voxel_grid_rgb_diff) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2world[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2world[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2world[2 * 4 + 3];
    float pt_cam_x = cam2world[0 * 4 + 0] * tmp_pt[0] + cam2world[1 * 4 + 0] * tmp_pt[1] + cam2world[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2world[0 * 4 + 1] * tmp_pt[0] + cam2world[1 * 4 + 1] * tmp_pt[1] + cam2world[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2world[0 * 4 + 2] * tmp_pt[0] + cam2world[1 * 4 + 2] * tmp_pt[1] + cam2world[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];
    unsigned char rgb_val[3] = {0};
    rgb_val[0] = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x * 3 + 2];
    rgb_val[1] = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x * 3 + 1];
    rgb_val[2] = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x * 3];

    if (depth_val <= 0 || depth_val > 6)
      continue;

//    float diff = (depth_val - pt_cam_z) * sqrtf(1 + powf((pt_cam_x / pt_cam_z), 2) + powf((pt_cam_y / pt_cam_z), 2));
    float diff = depth_val - pt_cam_z;
    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;

    if((rgb_val[0] == 0)&&(rgb_val[1] == 0)&&(rgb_val[2] == 0))
      continue;

    if(voxel_grid_rgb_weight[volume_idx] == 0)
    {
      voxel_grid_rgb[volume_idx * 3] = rgb_val[0];
      voxel_grid_rgb[volume_idx * 3 + 1] = rgb_val[1];
      voxel_grid_rgb[volume_idx * 3 + 2] = rgb_val[2];
      voxel_grid_rgb_weight[volume_idx] = voxel_grid_rgb_weight[volume_idx] + 1.0f; 
      continue;
    }
    float rgb_weight_old = voxel_grid_rgb_weight[volume_idx];
    float rgb_weight_new = rgb_weight_old + 1.0f;
    voxel_grid_rgb_weight[volume_idx] = rgb_weight_new;
    float rgb_diff = std::abs(voxel_grid_rgb[volume_idx * 3] - rgb_val[0]) + 
                     std::abs(voxel_grid_rgb[volume_idx * 3 + 1] - rgb_val[1]) + 
                     std::abs(voxel_grid_rgb[volume_idx * 3 + 2] - rgb_val[2]);
    voxel_grid_rgb_diff[volume_idx] = (voxel_grid_rgb_diff[volume_idx] * rgb_weight_old + rgb_diff) / rgb_weight_new;
    // std::cout<<"rgb_weight:"<<rgb_weight_new<<" rgb_diff:"<<voxel_grid_rgb_diff[volume_idx]<<std::endl;
    if(voxel_grid_rgb_diff[volume_idx]>90)
    {
      voxel_grid_rgb[volume_idx * 3] = rgb_val[0];
      voxel_grid_rgb[volume_idx * 3 + 1] = rgb_val[1];
      voxel_grid_rgb[volume_idx * 3 + 2] = rgb_val[2];
    }

  }
}

// Compute surface points from TSDF voxel grid and save points to point cloud file
void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                     float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                     float * voxel_grid_TSDF, float * voxel_grid_weight, unsigned char * voxel_grid_rgb,
                                     float tsdf_thresh, float weight_thresh) {

  // Count total number of points in point cloud
  int num_pts = 0;
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
      num_pts++;

  // Create header for .ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  // fprintf(fp, "format ascii 1.0\n");
  fprintf(fp, "element vertex %d\n", num_pts);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
      int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
      int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

      // Convert voxel indices to float, and save coordinates to ply file
      float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
      float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
      float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
      unsigned char pt_base_r = voxel_grid_rgb[i * 3];
      unsigned char pt_base_g = voxel_grid_rgb[i * 3 + 1];
      unsigned char pt_base_b = voxel_grid_rgb[i * 3 + 2];
      fwrite(&pt_base_x, sizeof(float), 1, fp);
      fwrite(&pt_base_y, sizeof(float), 1, fp);
      fwrite(&pt_base_z, sizeof(float), 1, fp);
      fwrite(&pt_base_r, sizeof(unsigned char), 1, fp);
      fwrite(&pt_base_g, sizeof(unsigned char), 1, fp);
      fwrite(&pt_base_b, sizeof(unsigned char), 1, fp);
      // if((pt_base_r<50)&&(pt_base_g<50)&&(pt_base_b<50))
      // {
      //   std::cout<<"r:"<<(int)pt_base_r;
      //   std::cout<<" g:"<<(int)pt_base_g;
      //   std::cout<<" b:"<<(int)pt_base_b<<std::endl;
      // }
      // std::cout<<"No: i"<<i<<" r:"<<(int)pt_base_r<<" g:"<< (int)pt_base_g<<" b:"<<(int)pt_base_r<<std::endl;
    }
  }
  fclose(fp);
}

// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}

std::vector<float> Matrix2Array(cv::Matx33f input, int M, int N)
{
  std::vector<float> array;
  for (int i = 0; i < M; i++)
  {
    for(int j=0;j<N;j++)
    {
        float tmp;
        tmp = input(i,j);
        array.push_back(tmp);
    }
  }
  return array;
}

std::vector<float> Matrix2Array(Eigen::Matrix4f input, int M, int N)
{
  std::vector<float> array;
  for (int i = 0; i < M; i++)
  {
    for(int j=0;j<N;j++)
    {
        float tmp;
        tmp = input(i,j);
        array.push_back(tmp);
    }
  }
  return array;
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth, int depth_scale) {
  cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / depth_scale;
      if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
        depth[r * W + c] = 0;
    }
}

void ReadDepth(cv::Mat depth_mat, int H, int W, float * depth, int depth_scale)
{
  if (depth_mat.empty())
  {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
    {
      depth[r * W + c] = (float)(depth_mat.at<float>(r, c))/depth_scale;
      //std::cout<<"depth:"<<depth_mat.at<float>(r, c)<<std::endl;
      if ((depth[r * W + c] > 6.0f)||(depth[r * W + c]!= depth[r * W + c])) // Only consider depth < 6m and remove "nan"
        depth[r * W + c] = 0;
    }
}

void ReadRGB(cv::Mat rgb_mat, int H, int W, unsigned char * rgb)
{
  if (rgb_mat.empty())
  {
    std::cout << "Error: rgb image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
    {
      for(int channel = 0; channel < 3; channel++)
      {
        rgb[r * W * 3 + c * 3 + channel] = rgb_mat.ptr<uchar>(r)[c*3+channel];
      }
    }
}

// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]) {
  mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
  mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
  mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
  mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

  mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
  mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
  mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
  mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

  mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
  mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;
  inv[0] = m[5]  * m[10] * m[15] -
           m[5]  * m[11] * m[14] -
           m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] -
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] +
           m[4]  * m[11] * m[14] +
           m[8]  * m[6]  * m[15] -
           m[8]  * m[7]  * m[14] -
           m[12] * m[6]  * m[11] +
           m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] -
           m[4]  * m[11] * m[13] -
           m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] +
            m[4]  * m[10] * m[13] +
            m[8]  * m[5] * m[14] -
            m[8]  * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] +
           m[1]  * m[11] * m[14] +
           m[9]  * m[2] * m[15] -
           m[9]  * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] -
           m[0]  * m[11] * m[14] -
           m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] +
           m[0]  * m[11] * m[13] +
           m[8]  * m[1] * m[15] -
           m[8]  * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] -
            m[0]  * m[10] * m[13] -
            m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] -
           m[1]  * m[7] * m[14] -
           m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] +
           m[0]  * m[7] * m[14] +
           m[4]  * m[2] * m[15] -
           m[4]  * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] -
            m[0]  * m[7] * m[13] -
            m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] +
            m[0]  * m[6] * m[13] +
            m[4]  * m[1] * m[14] -
            m[4]  * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

void FatalError(const int lineNumber) {
  std::cerr << "FatalError";
  if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
  std::cerr << ". Program Terminated." << std::endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
    FatalError();
  }
}

void RunKernal(float * cam_K, float * cam2world, float * depth_im, unsigned char * rgb_im,
                int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                float * voxel_grid_TSDF, float * voxel_grid_weight,
                unsigned char * voxel_grid_rgb, float * voxel_grid_rgb_weight, float * voxel_grid_rgb_diff)
{
    Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(cam_K, cam2world, depth_im, rgb_im,
                                                     im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                     voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                     voxel_grid_TSDF, voxel_grid_weight, voxel_grid_rgb, voxel_grid_rgb_weight, voxel_grid_rgb_diff);
}


