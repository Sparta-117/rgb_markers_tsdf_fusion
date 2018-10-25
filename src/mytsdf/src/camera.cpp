#include "camera.h"

namespace buildmodel
{

Camera::Camera(string camera_config_filename): _camera_config_filename(camera_config_filename)
{
    cout << "camera init" << endl;
    Config* myconfig = Config::getInstancePtr();
    myconfig->setParameterFile(camera_config_filename);
    cout << "camera init success" << endl;
//    _fx = myconfig->get<float>("camera.fx");
//    _fy = myconfig->get<float>("camera.fy");
//    _cx = myconfig->get<float>("camera.cx");
//    _cy = myconfig->get<float>("camera.cy");
//    _depth_scale = myconfig->get<float>("camera.depth_scale");
//    _dispara1 = myconfig->get<float>("camera.dispara1");
//    _dispara2 = myconfig->get<float>("camera.dispara2");
//    _dispara3 = myconfig->get<float>("camera.dispara3");
//    _dispara4 = myconfig->get<float>("camera.dispara4");

    _fx = 615.7;
    _fy = 615.7;
    _cx = 314.4;
    _cy = 238.9;

    //     _fx = 619.395;
    // _fy = 619.395;
    // _cx = 311.61;
    // _cy = 238.197;




    _depth_scale = 1;

    _dispara1 = 0;
    _dispara2 = 0;
    _dispara3 = 0;
    _dispara4 = 0;

    _intrinsic_matrix << _fx,0,_cx,0,
                        0,_fy,_cy,0,
                        0,0,1,0;
    _camera_intrinsic_matrix  <<_fx, 0, _cx,
                                0, _fy, _cy,
                                0, 0, 1;
    _distParam <<_dispara1, _dispara2, _dispara3, _dispara4;

    for (int i = 0; i < 3; i++)
    {
      for(int j=0;j<3;j++)
      {
          float tmp;
          tmp = _intrinsic_matrix(i,j);
          _camera_intrinsic_array.push_back(tmp);
      }
    }

}

Vector3d Camera::world2camera ( const Vector3d& p_w, const SE3& T_c_w )
{
    return T_c_w*p_w;
}

Vector3d Camera::camera2world ( const Vector3d& p_c, const SE3& T_c_w )
{
    return T_c_w.inverse() *p_c;
}

Vector2d Camera::camera2pixel ( const Vector3d& p_c )
{
    return Vector2d (
               _fx * p_c ( 0,0 ) / p_c ( 2,0 ) + _cx,
               _fy * p_c ( 1,0 ) / p_c ( 2,0 ) + _cy
           );
}

Vector3d Camera::pixel2camera ( const Vector2d& p_p, double depth )
{
    return Vector3d (
               ( p_p ( 0,0 )-_cx ) *depth/_fx,
               ( p_p ( 1,0 )-_cy ) *depth/_fy,
               depth
           );
}

Vector2d Camera::world2pixel ( const Vector3d& p_w, const SE3& T_c_w )
{
    return camera2pixel ( world2camera(p_w, T_c_w) );
}

Vector3d Camera::pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth )
{
    return camera2world ( pixel2camera ( p_p, depth ), T_c_w );
}

//std::vector<float> Camera::matrix2array(cv::Mat* input, int M, int N)
//{
//  std::vector<float> matrix;
//  for (int i = 0; i < M; i++)
//  {
//    for(int j=0;j<N;j++)
//    {
//        float tmp;
//        tmp = input->at<uchar>(i,j);
//        matrix.push_back(tmp);
//    }
//  }
//  return matrix;
//}

}
