#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
	InitialEXRotation();
    /*外参数求解:采用SVD分解，最小奇异值对应的向量即为我们的结果，
    在足够多的旋转运动中，我们可以很好的估计出相对旋转 [公式] ,这时 [公式] 对应一个准确解，且其零空间的秩为1。
    但是在校准的过程中，某些轴向上可能存在退化运动(如匀速运动)，这时 [公式] 的零空间的秩会大于1。
    判断条件就是 [公式] 的第二小的奇异值是否大于某个阈值，若大于则其零空间的秩为1，反之秩大于1，相对旋转) [公式] 的精度不够，校准不成功*/
    
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);

    /*对于任意一个 [公式] ，可以得到两个可能的 [公式] 和 [公式] 与之对应*/
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;

    vector< Matrix3d > Rc;
    vector< Matrix3d > Rimu;
    vector< Matrix3d > Rc_g;
    Matrix3d ric;
};


