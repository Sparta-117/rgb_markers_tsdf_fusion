#ifndef CAMERA_H
#define CAMERA_H

#include "common_include.h"
#include "config.h"

using namespace cv;
using namespace std;

namespace buildmodel
{
// Pinhole RGBD camera model
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    float   _fx, _fy, _cx, _cy, _depth_scale;
    float   _dispara1, _dispara2, _dispara3, _dispara4;

    Camera(string _camera_config_filename);
//    Camera ( float fx, float fy, float cx, float cy, float depth_scale=0 ) :
//        _fx ( fx ), _fy ( fy ), _cx ( cx ), _cy ( cy ), _depth_scale ( depth_scale )
//    {
//        _intrinsic_matrix << _fx,0,_cx,0,
//                            0,_fy,_cy,0,
//                            0,0,1,0;
//        _camera_intrinsic_matrix << _fx, 0, _cx,
//                                    0, _fy, _cy,
//                                    0, 0, 1
//                                    ;
//        _distParam << 0,0,0,0;
//        for (int i = 0; i < 3; i++)
//        {
//          for(int j=0;j<3;j++)
//          {
//              float tmp;
//              tmp = _intrinsic_matrix(i,j);
//              _camera_intrinsic_array.push_back(tmp);
//          }
//        }
//    }


    //Member Variables
    std::string _camera_config_filename;
    Eigen::Matrix<float,3,4> _intrinsic_matrix;
    cv::Matx33f _camera_intrinsic_matrix;
    std::vector<float> _camera_intrinsic_array;
    cv::Vec4f _distParam;

    // coordinate transform: world, camera, pixel
    Vector3d world2camera( const Vector3d& p_w, const SE3& T_c_w );
    Vector3d camera2world( const Vector3d& p_c, const SE3& T_c_w );
    Vector2d camera2pixel( const Vector3d& p_c );
    Vector3d pixel2camera( const Vector2d& p_p, double depth=1 );
    Vector3d pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth=1 );
    Vector2d world2pixel ( const Vector3d& p_w, const SE3& T_c_w );

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //此处添加宏定义
};

}
#endif // CAMERA_H
