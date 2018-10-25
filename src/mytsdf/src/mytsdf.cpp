#include "pangolin/pangolin.h"
#include "common_include.h"
#include "qrplane.h"
#include "config.h"
#include "camera.h"
//#include "utils.hpp"
//#include "tsdf_gpu.hpp"
#include "tsdf_gpu.cuh"
//#include "kinect_srv/rgbd_image.h"
// #include "realsense_msgs/realsense_msgs.h"
#include "rgbd_srv/rgbd.h"


using namespace std;
using namespace cv;

#define numberOfQRCodeOnSide 6

int main(int argc,char *argv[])
{
    Mat rgb_image;
    Mat rgb_tmp;
    Mat rgb_render;
    Mat depth_image;
    Mat depth_tmp;
    int index = 0;
    int depthScale = 1;
    //vector< int > id_list_pre;
    //Eigen::Matrix4f reference_pose;
    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr final_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

    //config file of camera
    std::string config_filename = "/home/mzm/mytsdf-fusion/src/mytsdf/kinect_config.yaml";
    buildmodel::Camera::Ptr camera1 (new buildmodel::Camera(config_filename));
    //buildmodel::QRPlane::Ptr plane_pre;
    //buildmodel::QRPlane::Ptr plane_last;

    float cam_K[3 * 3];
    float base2world[4 * 4];
    float base2world_inv[16] = {0};
    float cam2base[4 * 4];
    float cam2world[4 * 4];
    int im_width = 640;
    int im_height = 480;
    float depth_im[im_height * im_width];
    unsigned char rgb_im[im_height * im_width * 3];

    std::vector<float> cam_inK = Matrix2Array(camera1->_camera_intrinsic_matrix,3,3);
    std::copy(cam_inK.begin(), cam_inK.end(), cam_K);

    // Voxel grid parameters (change these to change voxel grid resolution, etc.)
    float voxel_grid_origin_x = -0.12f; // Location of voxel grid origin in base frame camera coordinates
    float voxel_grid_origin_y = -0.12f;
    float voxel_grid_origin_z = 0.0f;
    float voxel_size = 0.001f;
    float trunc_margin = voxel_size * 5;
    int voxel_grid_dim_x = 240;
    int voxel_grid_dim_y = 240;
    int voxel_grid_dim_z = 240;

    // Initialize voxel grid
    float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    unsigned char * voxel_grid_rgb = new unsigned char[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3];
    float * voxel_grid_rgb_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    float * voxel_grid_rgb_diff = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
      voxel_grid_TSDF[i] = 1.0f;
    memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    memset(voxel_grid_rgb, 0, sizeof(unsigned char) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3);
    memset(voxel_grid_rgb_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    memset(voxel_grid_rgb_diff, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

    // Load variables to GPU memory
    float * gpu_voxel_grid_TSDF;
    float * gpu_voxel_grid_weight;
    unsigned char * gpu_voxel_grid_rgb;
    float * gpu_voxel_grid_rgb_weight;
    float * gpu_voxel_grid_rgb_diff;
    cudaMalloc((void**)&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc((void**)&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc((void**)&gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&gpu_voxel_grid_rgb_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc((void**)&gpu_voxel_grid_rgb_diff, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_rgb, voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_rgb_weight, voxel_grid_rgb_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_rgb_diff, voxel_grid_rgb_diff, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    float * gpu_cam_K;
    float * gpu_cam2world;
    float * gpu_depth_im;
    unsigned char * gpu_rgb_im;
    cudaMalloc((void**)&gpu_cam_K, 3 * 3 * sizeof(float));
    cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_cam2world, 4 * 4 * sizeof(float));
    cudaMalloc((void**)&gpu_depth_im, im_height * im_width * sizeof(float));
    cudaMalloc((void**)&gpu_rgb_im, im_height * im_width * 3 * sizeof(unsigned char));
    checkCUDA(__LINE__, cudaGetLastError());

    //Pangolin
    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main",1280,980);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(0.35,0.35,0.3, 0,0,0, pangolin::AxisZ)
    );
    const int UI_WIDTH = 200;
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> save_button("ui.Save Model",false,false);

    // //定义图片面板
    // pangolin::View& rgb_show = pangolin::Display("rgb")
    //   .SetBounds(1,0,0,1,640.0f/480.0f)
    //   .SetLock(pangolin::LockRight, pangolin::LockBottom);

    pangolin::View& d_img1 = pangolin::Display("rgb")
        .SetAspect(640.0f/480.0f);
    pangolin::View& d_img2 = pangolin::Display("depth")
        .SetAspect(640.0f/480.0f);
    pangolin::View& d_img3 = pangolin::Display("edge")
        .SetAspect(640.0f/480.0f);

    // LayoutEqual is an EXPERIMENTAL feature - it requires that all sub-displays
    // share the same aspect ratio, placing them in a raster fasion in the
    // // viewport so as to maximise display size.
    pangolin::Display("multi")
        .SetBounds(0, 1 / 4.0f, pangolin::Attach::Pix(UI_WIDTH) , pangolin::Attach::Pix(1000))
        .SetLayout(pangolin::LayoutEqualHorizontal)
        .AddDisplay(d_img1)
        .AddDisplay(d_img2)
        .AddDisplay(d_img3);
    //Image texture
    pangolin::GlTexture rgb_imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
    pangolin::GlTexture depth_imageTexture(640,480,GL_DEPTH_COMPONENT,false,0,GL_DEPTH_COMPONENT,GL_FLOAT);
    pangolin::GlTexture edge_imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    //ros node
    ros::init(argc, argv, "mytsdf");
    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<rgbd_srv::rgbd> ( "realsense2_server" );
    rgbd_srv::rgbd srv;
    srv.request.start = true;
    sensor_msgs::Image msg_rgb;
    sensor_msgs::Image msg_depth;
    ros::Rate loop_rate(200);
    ROS_INFO("Start TSDF!");

    while((ros::ok())&&(!pangolin::ShouldQuit()))
    {
        if (client.call(srv))
        {
            try
            {
              msg_rgb = srv.response.rgb_image;
              msg_depth = srv.response.depth_image;
              rgb_image = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::TYPE_8UC3)->image;
              depth_image = cv_bridge::toCvCopy(msg_depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
              normalize(depth_image,depth_tmp,255,0,NORM_MINMAX);
              depth_tmp.convertTo(depth_tmp, CV_8UC1, 1.0);
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception: %s", e.what());
              return 1;
            }
            IplImage ipl_rgb_image = rgb_image;
            cvConvertImage(&ipl_rgb_image , &ipl_rgb_image , CV_CVTIMG_SWAP_RB);
            depth_image.convertTo(depth_image,CV_32F);
        }

        if( !rgb_image.data )
        {
            printf( " No image data \n " );
            return -1;
        }

        //pangolin
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f,1.0f,1.0f,1.0f);    

        rgb_image.copyTo(rgb_tmp);
        rgb_image.copyTo(rgb_render);

        buildmodel::QRPlane::Ptr plane1 (new buildmodel::QRPlane(numberOfQRCodeOnSide,
                                                                  camera1->_camera_intrinsic_matrix,
                                                                  camera1->_distParam));
        plane1->_rgb_image = rgb_image;
        plane1->_depth_image = depth_image;
        plane1->detectQRCode();
        if(plane1->whetherDectectQRCode())
        {
            plane1->GetTransfromCameraToWorld();
            plane1->DrawAxis(rgb_tmp);
            plane1->RenderRGBImage(rgb_render);
            //plane1->DrawCube(rgb_tmp);
            //plane1->CutCubeInDepthImageAndPointsCloud(result_depth_image,cube_points_cloud);
            index++;
        }
        else
        {
            cout<<"Not find QR code!"<<endl;
            continue;
        }
        imshow("rgb_tmp",rgb_tmp);
        imshow("depth",depth_image);
        imshow("rgb_render",rgb_render);
        waitKey(1);
        cout<<"camera2world:"<<plane1->_transform_camera2world<<endl;


        if(index == 1)
        {
        std::vector<float> base2world_vec = Matrix2Array(plane1->_transform_camera2world, 4, 4);
        std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);
        // Invert base frame camera pose to get world-to-base frame transform
        invert_matrix(base2world, base2world_inv);
        }

        // Read current frame depth
        ReadDepth(depth_image, im_height, im_width, depth_im, depthScale);

        // Read current frame rgb
        ReadRGB(rgb_render, im_height, im_width, rgb_im);

        // Read base frame camera pose
        std::vector<float> cam2world_vec = Matrix2Array(plane1->_transform_camera2world, 4, 4);
        std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

        // Compute relative camera pose (camera-to-base frame)
        multiply_matrix(base2world_inv, cam2world, cam2base);

        cudaMemcpy(gpu_cam2world, cam2world, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
//        cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_depth_im, depth_im, im_height * im_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_rgb_im, rgb_im, im_height * im_width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

        cout << "Fusing: " << index << endl; 

        RunKernal(gpu_cam_K, gpu_cam2world, gpu_depth_im, gpu_rgb_im,
                  im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                  voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                  gpu_voxel_grid_TSDF, gpu_voxel_grid_weight, gpu_voxel_grid_rgb, gpu_voxel_grid_rgb_weight, gpu_voxel_grid_rgb_diff);

        // Load TSDF voxel grid from GPU to CPU memory
        cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(voxel_grid_rgb, gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        // cudaMemcpy(voxel_grid_rgb_weight, gpu_voxel_grid_rgb_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(voxel_grid_rgb_diff, gpu_voxel_grid_rgb_diff, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDA(__LINE__, cudaGetLastError());

        d_cam.Activate(s_cam);
        glPointSize(2);
        glBegin(GL_POINTS);

        #pragma omp parallel for
        for(int i=0; i<voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z;i++)
        {
            // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
            if (std::abs(voxel_grid_TSDF[i]) < 0.15f && voxel_grid_weight[i] > 15.0f) 
            {
                // Compute voxel indices in int for higher positive number range
                int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
                int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
                int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

                // Convert voxel indices to float, and save coordinates to ply file
                float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
                float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
                float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
                float pt_base_r = (float)voxel_grid_rgb[i * 3]/255;
                float pt_base_g = (float)voxel_grid_rgb[i * 3 + 1]/255;
                float pt_base_b = (float)voxel_grid_rgb[i * 3 + 2]/255;
                glColor3f(pt_base_r, pt_base_g, pt_base_b);
                glVertex3d(pt_base_x, pt_base_y, pt_base_z);
            } 
        }
        glEnd();

        if(pangolin::Pushed(save_button))
        {
            // Load TSDF voxel grid from GPU to CPU memory
            cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(voxel_grid_rgb, gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cudaMemcpy(voxel_grid_rgb_weight, gpu_voxel_grid_rgb_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(voxel_grid_rgb_diff, gpu_voxel_grid_rgb_diff, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
            checkCUDA(__LINE__, cudaGetLastError());

            // for(int i=0; i<voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z;i++)
            // {
            //     cout<<"rgb weight: "<<voxel_grid_rgb_weight[i]<<"rgb diff:"<<voxel_grid_rgb_diff[i]<<endl;
            // }
            // Compute surface points from TSDF voxel grid and save to point cloud .ply file
            std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
        //    std::cout<<"voxel_grid_dim_x:"<<voxel_grid_TSDF<<std::endl;
            SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                            voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                            voxel_grid_TSDF, voxel_grid_weight, voxel_grid_rgb, 0.15f, 15.0f);

            // Save TSDF voxel grid and its parameters to disk as binary file (float array)
            std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
            std::string voxel_grid_saveto_path = "tsdf.bin";
            std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
            float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
            float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
            float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
            outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
            outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
            outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
            outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
            outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
            outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
            outFile.write((char*)&voxel_size, sizeof(float));
            outFile.write((char*)&trunc_margin, sizeof(float));
            for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
            {
              // cout<<"No:"<<i<<" voxel:"<<(char*)&voxel_grid_TSDF[i]<<endl;
              outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
            }
            outFile.close();

        }

        //Set some random image data and upload to GPU
        // unsigned char * imageArray;
        // memcpy((unsigned char*)imageArray.begin(), (void*)rgb_render.data, rgb_render.total() * rgb_render.elemSize()); //TODO
        rgb_imageTexture.Upload(rgb_image.data,GL_BGR,GL_UNSIGNED_BYTE);
        depth_imageTexture.Upload(depth_image.data,GL_DEPTH_COMPONENT,GL_FLOAT);
        edge_imageTexture.Upload(rgb_render.data,GL_BGR,GL_UNSIGNED_BYTE);
        // Activate efficiently by object
        d_img1.Activate();
        glColor3f(1.0f,1.0f,1.0f);
        rgb_imageTexture.RenderToViewportFlipY();

        d_img2.Activate();
        glColor3f(1.0f,1.0f,1.0f);
        depth_imageTexture.RenderToViewportFlipY();

        d_img3.Activate();
        glColor3f(1.0f,1.0f,1.0f);
        edge_imageTexture.RenderToViewportFlipY();
        // Swap frames and Process Events
        pangolin::FinishFrame();

    }
//     // Load TSDF voxel grid from GPU to CPU memory
//     cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(voxel_grid_rgb, gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//     cudaMemcpy(voxel_grid_rgb_weight, gpu_voxel_grid_rgb_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(voxel_grid_rgb_diff, gpu_voxel_grid_rgb_diff, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
//     checkCUDA(__LINE__, cudaGetLastError());

//     // for(int i=0; i<voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z;i++)
//     // {
//     //     cout<<"rgb weight: "<<voxel_grid_rgb_weight[i]<<"rgb diff:"<<voxel_grid_rgb_diff[i]<<endl;
//     // }
//     // Compute surface points from TSDF voxel grid and save to point cloud .ply file
//     std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
// //    std::cout<<"voxel_grid_dim_x:"<<voxel_grid_TSDF<<std::endl;
//     SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
//                                     voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
//                                     voxel_grid_TSDF, voxel_grid_weight, voxel_grid_rgb, 0.15f, 10.0f);

//     // Save TSDF voxel grid and its parameters to disk as binary file (float array)
//     std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
//     std::string voxel_grid_saveto_path = "tsdf.bin";
//     std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
//     float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
//     float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
//     float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
//     outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
//     outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
//     outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
//     outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
//     outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
//     outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
//     outFile.write((char*)&voxel_size, sizeof(float));
//     outFile.write((char*)&trunc_margin, sizeof(float));
//     for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
//     {
//       // cout<<"No:"<<i<<" voxel:"<<(char*)&voxel_grid_TSDF[i]<<endl;
//       outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
//     }
//     outFile.close();

    return 0;
}
