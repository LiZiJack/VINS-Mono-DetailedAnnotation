#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

/*
网上：在这里，为什么要多此一举构造一个sfm_f而不是直接使用f_manager呢？ 我的理解，
是因为f_manager的信息量大于SfM所需的信息量(f_manager包含了大量的像素信息)，
而且不同的数据结构是为了不同的业务服务的，所以在这里作者专门为SfM设计了一个全新的数据结构sfm_f，专业化服务。

*/
struct SFMFeature
{
    bool state;//特征点的状态（是否被三角化）
    int id;//特征点id
    vector<pair<int,Vector2d>> observation;//所有观测到该特征点的图像帧ID和图像坐标
    double position[3];//3d坐标   在帧L下的空间坐标
    double depth;//深度
};

//  camera_R, //L帧图像到i帧图像R
//  camera_T, //L帧图像到i帧图像平移
//  point, //3d投标点空间坐标值  注意point在L帧坐标系下
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;//前端跟踪到的图像归一化平面坐标   前端跟踪时用liftProjective转的
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};