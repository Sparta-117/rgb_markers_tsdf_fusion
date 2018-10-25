#include "qrplane.h"

using namespace std;
using namespace cv;

namespace buildmodel
{
QRPlane::QRPlane()
{}
QRPlane::QRPlane(int numberOfQROnSide,cv::Matx33f camera_intrinsic_matrix,cv::Vec4f distParam):
    _numberOfQROnSide(numberOfQROnSide),
    _camera_intrinsic_matrix(camera_intrinsic_matrix),
    _distParam(distParam)
{
    if(_numberOfQROnSide == 10)
    {
        TEN_x_TEN ten_x_ten;
        _qr = ten_x_ten;
    }
    if(_numberOfQROnSide == 6)
    {
        SIX_x_SIX six_x_six;
        _qr = six_x_six;
    }
    _fx=_camera_intrinsic_matrix(0,0);
    _fy=_camera_intrinsic_matrix(1,1);
    _cx=_camera_intrinsic_matrix(0,2);
    _cy=_camera_intrinsic_matrix(1,2);

    _transform_world2camera = Eigen::Matrix4f::Identity();
    _transform_camera2world = Eigen::Matrix4f::Identity();
}

void QRPlane::detectQRCode()
{
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    aruco::detectMarkers(_rgb_image, _qr.dictionary, _corners, _ids, detectorParams, _rejected);
}

bool QRPlane::whetherDectectQRCode()
{
    if (_ids.size() <= 0)
    {
        return 0;
    }
    else
    {
        cout<<"id size:"<<_ids.size()<<endl;
        return 1;
    }
}

void QRPlane::GetTransfromCameraToWorld()
{
    double s = _qr.QR_TAG_SIZE/2;
    for (int i = 0; i < _ids.size(); i++)
    {
//        cout<<"ids[i]:"<<_ids[i]<<endl;
        float dx = _qr.basePositions[_ids[i]](0);
        float dy = _qr.basePositions[_ids[i]](1);
//        cout<<"dx"<<dx<<endl;
//        cout<<"dy"<<dy<<endl;
//        cout<<"s"<<s<<endl;
        _CornersInWorldCor.push_back(cv::Point3f(-s + dx, s + dy, 0));
        _CornersInWorldCor.push_back(cv::Point3f(s + dx, s + dy, 0));
        _CornersInWorldCor.push_back(cv::Point3f(s + dx, -s + dy, 0));
        _CornersInWorldCor.push_back(cv::Point3f(-s + dx, -s + dy, 0));

        float p1x = _corners[i][0].x;
        float p1y = _corners[i][0].y;
        float p2x = _corners[i][1].x;
        float p2y = _corners[i][1].y;
        float p3x = _corners[i][2].x;
        float p3y = _corners[i][2].y;
        float p4x = _corners[i][3].x;
        float p4y = _corners[i][3].y;
        _CornersInImageCor.push_back(cv::Point2f(p1x, p1y));
        _CornersInImageCor.push_back(cv::Point2f(p2x, p2y));
        _CornersInImageCor.push_back(cv::Point2f(p3x, p3y));
        _CornersInImageCor.push_back(cv::Point2f(p4x, p4y));

    }

    solvePnP(_CornersInWorldCor, _CornersInImageCor,  _camera_intrinsic_matrix, _distParam, _rvec, _tvec);

    cv::Matx33d r;
    cv::Rodrigues(_rvec, r);
    _rotation<< r(0, 0), r(0, 1), r(0, 2),
                r(1, 0), r(1, 1), r(1, 2),
                r(2, 0), r(2, 1), r(2, 2);
    _translation<<  (float)_tvec.at<double>(0,0),(float)_tvec.at<double>(0,1),(float)_tvec.at<double>(0,2);

    _transform_world2camera (0,0) = r(0, 0);
    _transform_world2camera (0,1) = r(0, 1);
    _transform_world2camera (0,2) = r(0, 2);
    _transform_world2camera (1,0) = r(1, 0);
    _transform_world2camera (1,1) = r(1, 1);
    _transform_world2camera (1,2) = r(1, 2);
    _transform_world2camera (2,0) = r(2, 0);
    _transform_world2camera (2,1) = r(2, 1);
    _transform_world2camera (2,2) = r(2, 2);
    _transform_world2camera (0,3) = (float)_tvec.at<double>(0,0);
    _transform_world2camera (1,3) = (float)_tvec.at<double>(0,1);
    _transform_world2camera (2,3) = (float)_tvec.at<double>(0,2);

    _transform_camera2world = _transform_world2camera.inverse();

    for (int i = 0; i < _ids.size(); i++)
    {
//        cout<<"ids[i]:"<<_ids[i]<<endl;
        SingleQRCorner::Ptr singleqrcorner(new SingleQRCorner);
        singleqrcorner->id = _ids[i];
        float dx = _qr.basePositions[_ids[i]](0);
        float dy = _qr.basePositions[_ids[i]](1);

        singleqrcorner->CornerInWorldCor[0] << -s + dx, s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[1] << s + dx, s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[2] << s + dx, -s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[3] << -s + dx, -s + dy, 0, 1;

        float p1x = _corners[i][0].x;
        float p1y = _corners[i][0].y;
        float p2x = _corners[i][1].x;
        float p2y = _corners[i][1].y;
        float p3x = _corners[i][2].x;
        float p3y = _corners[i][2].y;
        float p4x = _corners[i][3].x;
        float p4y = _corners[i][3].y;

        singleqrcorner->CornerInImageCor[0] << p1x, p1y;
        singleqrcorner->CornerInImageCor[1] << p2x, p2y;
        singleqrcorner->CornerInImageCor[2] << p3x, p3y;
        singleqrcorner->CornerInImageCor[3] << p4x, p4y;

        singleqrcorner->CornerInCameraCor[0] = _transform_world2camera*singleqrcorner->CornerInWorldCor[0];
        singleqrcorner->CornerInCameraCor[1] = _transform_world2camera*singleqrcorner->CornerInWorldCor[1];
        singleqrcorner->CornerInCameraCor[2] = _transform_world2camera*singleqrcorner->CornerInWorldCor[2];
        singleqrcorner->CornerInCameraCor[3] = _transform_world2camera*singleqrcorner->CornerInWorldCor[3];

        singleqrcorner->CornerIndepthImage[0] = _depth_image.at<float>(p1y,p1x);
        singleqrcorner->CornerIndepthImage[1] = _depth_image.at<float>(p2y,p2x);
        singleqrcorner->CornerIndepthImage[2] = _depth_image.at<float>(p3y,p3x);
        singleqrcorner->CornerIndepthImage[3] = _depth_image.at<float>(p4y,p4x);

//        cout<<"CornerInCameraCor:"<<singleqrcorner->CornerInCameraCor[0]<<endl;
        _QRCornersList.push_back(*singleqrcorner);
    }


//        cout<<"transform_world2camera:"<<_transform_world2camera<<endl;
//        cout<<"transform_world2camera_inverse:"<<_transform_world2camera_inverse<<endl;
//        cout<<"tvec:"<<_tvec<<endl;

}

void QRPlane::DrawAxis(Mat &tmp)
{
    cv::aruco::drawAxis(tmp, _camera_intrinsic_matrix,  _distParam, _rvec, _tvec, 0.2);
}

void QRPlane::DrawCube(Mat &tmp)
{
    Eigen::Matrix<float,3,1> image_square[8];
    Eigen::Matrix<float,3,4> intrinsic_matrix;
    intrinsic_matrix << _fx,0,_cx,0,
                        0,_fy,_cy,0,
                        0,0,1,0;
    for(int k=0;k<8;k++)
    {
        image_square[k] = intrinsic_matrix*_transform_world2camera*_qr.cubeVertexInWorld[k];
        _cubeVertexInImage[k].x = (float)image_square[k](0)/image_square[k](2);
        _cubeVertexInImage[k].y = (float)image_square[k](1)/image_square[k](2);
    }
    //draw cube
    line(tmp, _cubeVertexInImage[0], _cubeVertexInImage[1], Scalar(0, 255, 0), 3);
    line(tmp, _cubeVertexInImage[1], _cubeVertexInImage[2], Scalar(0, 255, 0), 3);
    line(tmp, _cubeVertexInImage[2], _cubeVertexInImage[3], Scalar(0, 255, 0), 3);
    line(tmp, _cubeVertexInImage[3], _cubeVertexInImage[0], Scalar(0, 255, 0), 3);

    line(tmp, _cubeVertexInImage[4], _cubeVertexInImage[5], Scalar(0, 0, 255), 3);
    line(tmp, _cubeVertexInImage[5], _cubeVertexInImage[6], Scalar(0, 0, 255), 3);
    line(tmp, _cubeVertexInImage[6], _cubeVertexInImage[7], Scalar(0, 0, 255), 3);
    line(tmp, _cubeVertexInImage[7], _cubeVertexInImage[4], Scalar(0, 0, 255), 3);

    line(tmp, _cubeVertexInImage[0], _cubeVertexInImage[4], Scalar(255, 255, 0), 3);
    line(tmp, _cubeVertexInImage[1], _cubeVertexInImage[5], Scalar(255, 255, 0), 3);
    line(tmp, _cubeVertexInImage[2], _cubeVertexInImage[6], Scalar(255, 255, 0), 3);
    line(tmp, _cubeVertexInImage[3], _cubeVertexInImage[7], Scalar(255, 255, 0), 3);
}

void QRPlane::CutCubeInDepthImage(Mat &result_depth_image)
{
    Point points[1][4];
    points[0][0] = _cubeVertexInImage[0];
    points[0][1] = _cubeVertexInImage[1];
    points[0][2] = _cubeVertexInImage[2];
    points[0][3] = _cubeVertexInImage[3];

    const Point* pt[1] = { points[0] };
    int npt[1] = {4};

    Mat roi_image = Mat::zeros(_depth_image.size(),CV_32FC1);

    polylines( roi_image, pt, npt, 1, 1, Scalar(255)) ;
    fillPoly( roi_image, pt, npt, 1, Scalar(255), 8);

    for (int r=0;r<_depth_image.rows;r++)
    {
        for (int c=0;c<_depth_image.cols;c++)
        {
            if(!(_depth_image.at<float>(r,c)>0))
            continue;

            if(roi_image.at<float>(r,c)>0)
            {
                result_depth_image.at<float>(r,c) = _depth_image.at<float>(r,c);
                continue;
            }

            double scene_z = double(_depth_image.at<float>(r,c));
            double scene_x = (c - _cx)*scene_z / _fx;
            double scene_y = (r - _cy)*scene_z / _fy;

            Eigen::Matrix<float,4,1> scene_point;
            scene_point << scene_x/1000,scene_y/1000,scene_z/1000,1;
            Eigen::Matrix<float,4,1> world_point;
            world_point = _transform_camera2world*scene_point;

            if( (abs(world_point(0))>_qr.cubeSide/2)||
                (abs(world_point(1))>_qr.cubeSide/2)||
                (world_point(2)>_qr.cubeHeight)||
                (world_point(2)<-0.005))
                continue;
            else
            {
                result_depth_image.at<float>(r,c) = _depth_image.at<float>(r,c);
//                    cout<<"point:"<<_depth_image.at<float>(r,c)<<endl;
//                    cout<<"z:"<<world_point(2)<<endl;
            }
        }
    }
}

void QRPlane::CutCubeInDepthImageAndPointsCloud(Mat &result_depth_image,pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cube_points_cloud)
{
    Point points[1][4];
    points[0][0] = _cubeVertexInImage[0];
    points[0][1] = _cubeVertexInImage[1];
    points[0][2] = _cubeVertexInImage[2];
    points[0][3] = _cubeVertexInImage[3];

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointXYZRGBA>);
    bool form_points_cloud;

    const Point* pt[1] = { points[0] };
    int npt[1] = {4};

    Mat roi_image = Mat::zeros(_depth_image.size(),CV_32FC1);

    polylines( roi_image, pt, npt, 1, 1, Scalar(255)) ;
    fillPoly( roi_image, pt, npt, 1, Scalar(255), 8);

    for (int r=0;r<_depth_image.rows;r++)
    {
        for (int c=0;c<_depth_image.cols;c++)
        {
            pcl::PointXYZRGBA p;
            if(!(_depth_image.at<float>(r,c)>0))
            continue;

            double scene_z = double(_depth_image.at<float>(r,c));
            double scene_x = (c - _cx)*scene_z / _fx;
            double scene_y = (r - _cy)*scene_z / _fy;

            if(roi_image.at<float>(r,c)>0)
            {
                result_depth_image.at<float>(r,c) = _depth_image.at<float>(r,c);
                p.x = 0.001*scene_x;
                p.y = 0.001*scene_y;
                p.z = 0.001*scene_z;
                p.b = _rgb_image.ptr<uchar>(r)[c*3];
                p.g = _rgb_image.ptr<uchar>(r)[c*3+1];
                p.r = _rgb_image.ptr<uchar>(r)[c*3+2];
                cloud_scene->points.push_back(p);
                continue;
            }

            Eigen::Matrix<float,4,1> scene_point;
            scene_point << scene_x/1000,scene_y/1000,scene_z/1000,1;
            Eigen::Matrix<float,4,1> world_point;
            world_point = _transform_camera2world*scene_point;

            if( (abs(world_point(0))>_qr.cubeSide/2)||
                (abs(world_point(1))>_qr.cubeSide/2)||
                (world_point(2)>_qr.cubeHeight)||
                (world_point(2)<-0.005))
                continue;
            else
            {
                result_depth_image.at<float>(r,c) = _depth_image.at<float>(r,c);
                p.x = 0.001*scene_x;
                p.y = 0.001*scene_y;
                p.z = 0.001*scene_z;
                p.b = _rgb_image.ptr<uchar>(r)[c*3];
                p.g = _rgb_image.ptr<uchar>(r)[c*3+1];
                p.r = _rgb_image.ptr<uchar>(r)[c*3+2];
                cloud_scene->points.push_back(p);
//                    cout<<"point:"<<_depth_image.at<float>(r,c)<<endl;
//                    cout<<"z:"<<world_point(2)<<endl;
            }
        }
    }
    cube_points_cloud = cloud_scene;
    _cube_cloud = cloud_scene;
}

//void QRPlane::CutCubeInPointsCloud()
//{
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointXYZRGBA>);
//    for (int r=0;r<_rgb_image.rows;r++)
//    {
//        for (int c=0;c<_rgb_image.cols;c++)
//        {
//            pcl::PointXYZRGBA p;

//            double scene_z = double(_depth_image.at<float>(r,c));
//            double scene_x = (c - _cx)*scene_z / _fx;
//            double scene_y = (r - _cy)*scene_z / _fy;
//            p.x = 0.001*scene_x;
//            p.y = 0.001*scene_y;
//            p.z = 0.001*scene_z;
//            p.b = _rgb_image.ptr<uchar>(r)[c*3];
//            p.g = _rgb_image.ptr<uchar>(r)[c*3+1];
//            p.r = _rgb_image.ptr<uchar>(r)[c*3+2];
//            cloud_scene->points.push_back(p);
//        }
//    }
//    _cube_cloud = cloud_scene;
//}

void QRPlane::AllQRCornersInCamCor()
{
    double s = _qr.QR_TAG_SIZE/2;
    Eigen::Matrix<float,3,4> intrinsic_matrix;
    intrinsic_matrix << _fx,0,_cx,0,
                        0,_fy,_cy,0,
                        0,0,1,0;

    for(int i=0;i<_qr.QRNumber;i++)
    {
        SingleQRCorner::Ptr singleqrcorner(new SingleQRCorner);
        singleqrcorner->id = i;
        float dx = _qr.basePositions[i](0);
        float dy = _qr.basePositions[i](1);

        singleqrcorner->CornerInWorldCor[0] << -s + dx, s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[1] << s + dx, s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[2] << s + dx, -s + dy, 0, 1;
        singleqrcorner->CornerInWorldCor[3] << -s + dx, -s + dy, 0, 1;

        Eigen::Vector3f pointInImage[4];
        pointInImage[0] = intrinsic_matrix*_transform_world2camera*singleqrcorner->CornerInWorldCor[0];
        pointInImage[1] = intrinsic_matrix*_transform_world2camera*singleqrcorner->CornerInWorldCor[1];
        pointInImage[2] = intrinsic_matrix*_transform_world2camera*singleqrcorner->CornerInWorldCor[2];
        pointInImage[3] = intrinsic_matrix*_transform_world2camera*singleqrcorner->CornerInWorldCor[3];

        float p1x = pointInImage[0](0)/pointInImage[0](2);
        float p1y = pointInImage[0](1)/pointInImage[0](2);
        float p2x = pointInImage[1](0)/pointInImage[1](2);
        float p2y = pointInImage[1](1)/pointInImage[1](2);
        float p3x = pointInImage[2](0)/pointInImage[2](2);
        float p3y = pointInImage[2](1)/pointInImage[2](2);
        float p4x = pointInImage[3](0)/pointInImage[3](2);
        float p4y = pointInImage[3](1)/pointInImage[3](2);

        singleqrcorner->CornerInImageCor[0] << p1x,p1y;
        singleqrcorner->CornerInImageCor[1] << p2x,p2y;
        singleqrcorner->CornerInImageCor[2] << p3x,p3y;
        singleqrcorner->CornerInImageCor[3] << p4x,p4y;

        singleqrcorner->CornerInCameraCor[0] = _transform_world2camera*singleqrcorner->CornerInWorldCor[0];
        singleqrcorner->CornerInCameraCor[1] = _transform_world2camera*singleqrcorner->CornerInWorldCor[1];
        singleqrcorner->CornerInCameraCor[2] = _transform_world2camera*singleqrcorner->CornerInWorldCor[2];
        singleqrcorner->CornerInCameraCor[3] = _transform_world2camera*singleqrcorner->CornerInWorldCor[3];

        singleqrcorner->CornerIndepthImage[0] = 0;
        singleqrcorner->CornerIndepthImage[1] = 0;
        singleqrcorner->CornerIndepthImage[2] = 0;
        singleqrcorner->CornerIndepthImage[3] = 0;

//        cout<<"CornerInCameraCor:"<<singleqrcorner->CornerInCameraCor[0]<<endl;
        _AllQRCornersList.push_back(*singleqrcorner);
    }
}

void QRPlane::SaveQRPlaneData(int index)
{
    char new_file_name[5]="0000";
    for(int j=3;j>=0;j--)
    {
    if(index>0)
    {
    new_file_name[j] ='0' + index%10;
    index/=10;
    }
    else new_file_name[j] = '0';
    }

    stringstream ss;
    string i2;
    ss<<new_file_name;
    ss>>i2;
    string filename_Pose = "/home/mzm/new_sr300_build_model/src/build_model2/data/Pose/" + i2 +".txt";
    string filename_CornerInCamCor = "/home/mzm/new_sr300_build_model/src/build_model2/data/CornerInCamCor/" + i2 +".txt";
    string filename_CornerInImageCor = "/home/mzm/new_sr300_build_model/src/build_model2/data/CornerInImageCor/" + i2 +".txt";
    string filename_PointsCloud = "/home/mzm/new_sr300_build_model/src/build_model2/data/PointsCloud/" + i2 +".ply";

    cout<<"Saving data!"<<endl;
    ofstream outfile;

    outfile.open(filename_CornerInCamCor);
//    outfile<<"CornersInCameraCor:"<<endl;
    outfile<<"num: "<<_AllQRCornersList.size()<<endl;
    for(int i=0;i<_AllQRCornersList.size();i++)
    {
        outfile<<"id: "<<_AllQRCornersList[i].id<<endl;
        for(int m=0;m<4;m++)
        {
            outfile<<_AllQRCornersList[i].CornerInCameraCor[m](0)<<" "
                   <<_AllQRCornersList[i].CornerInCameraCor[m](1)<<" "
                   <<_AllQRCornersList[i].CornerInCameraCor[m](2)<<" "
                   <<_AllQRCornersList[i].CornerInCameraCor[m](3);
            outfile<<endl;
        }
        outfile<<endl;

    }
    outfile.close();

    outfile.open(filename_CornerInImageCor);
//    outfile<<"CornersInImageCor:"<<endl;
    outfile<<"num: "<<_QRCornersList.size()<<endl;
    for(int i=0;i<_QRCornersList.size();i++)
    {
        outfile<<"id: "<<_QRCornersList[i].id<<endl;
        for(int m=0;m<4;m++)
        {
            outfile<<_QRCornersList[i].CornerInImageCor[m](0)<<" "
                   <<_QRCornersList[i].CornerInImageCor[m](1);
            outfile<<endl;
        }
    }
    outfile.close();

    outfile.open(filename_Pose);
//    outfile<<"PoseToWorld:"<<endl;
    outfile<<_transform_world2camera<<endl;
    outfile.close();

    pcl::PLYWriter writer;
    writer.write(filename_PointsCloud,*_cube_cloud);

}

void QRPlane::TransformPointsCloudCorToWorld(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cube_points_cloud)
{
    //Executing the transformation
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    // You can either apply transform_1 or transform_2; they are the same
    pcl::transformPointCloud (*cube_points_cloud, *transformed_cloud , _transform_camera2world);
    pcl::PLYWriter writer;
    writer.write("/home/mzm/new_sr300_build_model/src/build_model2/data/bottom.ply",*transformed_cloud );
}

void QRPlane::RenderRGBImage(Mat &render_rgb_image)
{
    Mat edge_mask = Mat::zeros(_depth_image.size(),CV_8UC1);
    for(int r = 0; r < edge_mask.rows; r++)
    {
        for(int c = 0; c<edge_mask.cols; c++)
        {
            if(!(_depth_image.ptr<float>(r)[c] > 0))
            {
                edge_mask.ptr<uchar>(r)[c] = 255;
                continue;
            }
            float p1 = _depth_image.ptr<float>(r-1)[c-1];
            float p2 = _depth_image.ptr<float>(r-1)[c];
            float p3 = _depth_image.ptr<float>(r-1)[c+1];
            float p4 = _depth_image.ptr<float>(r)[c-1];
            float p5 = _depth_image.ptr<float>(r)[c];
            float p6 = _depth_image.ptr<float>(r)[c+1];
            float p7 = _depth_image.ptr<float>(r+1)[c-1];
            float p8 = _depth_image.ptr<float>(r+1)[c];
            float p9 = _depth_image.ptr<float>(r+1)[c+1];

            float G0 = (p1+p2+p3)-(p7+p8+p9);
            float G1 = (p1+p4+p7)-(p3+p6+p9);

            // std::cout<<"G0:"<<G0<<" G1:"<<G1<<endl;

            // float pmax = max(p1,p2,p3,p4,p5,p6,p7,p8,p9);
            // float pmin = min(p1,p2,p3,p4,p5,p6,p7,p8,p9);
            // float pmean = (pmax+pmin)/2;
            if(max(G0,G1)>30)
            {
                edge_mask.ptr<uchar>(r)[c] = 255;
            }
        }
    }
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); 
    dilate(edge_mask, edge_mask, element);
    // imshow("edge_mask",edge_mask);
    // waitKey(1);
    for(int r = 0; r < render_rgb_image.rows; r++)
    {
        for(int c = 0; c<render_rgb_image.cols; c++)
        {
            if(edge_mask.ptr<uchar>(r)[c] == 255)
            {
                render_rgb_image.ptr<uchar>(r)[c*3] = 0;
                render_rgb_image.ptr<uchar>(r)[c*3+1] = 0;
                render_rgb_image.ptr<uchar>(r)[c*3+2] = 0;
            }
        }
    }


}

}

