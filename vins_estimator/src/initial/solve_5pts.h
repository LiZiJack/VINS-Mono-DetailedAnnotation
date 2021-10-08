#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
/*
数据结构: MotionEstimator m_estimator 它定义在initial/solve_5pts.h中，
这个类没有数据成员，只有函数功能。所以说，TODO 把这个h文件干掉，融到estimator.cpp里面去！ 

*/
class MotionEstimator
{
  public:

    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

  private:
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};


