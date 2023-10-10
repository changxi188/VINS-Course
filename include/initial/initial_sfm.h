#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

struct SFMFeature
{
    bool                                         state;
    int                                          id;
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    double                                       position[3];
    double                                       depth;
};

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v) : observed_u(observed_u), observed_v(observed_v)
    {
    }

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp         = p[0] / p[2];
        T yp         = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
            new ReprojectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};

class GlobalSFM
{
public:
    GlobalSFM();
    bool construct(const int frame_num, const int l, const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
                   Eigen::Quaterniond* Q, Eigen::Vector3d* T, std::vector<SFMFeature>& sfm_f,
                   std::map<int, Eigen::Vector3d>& sfm_tracked_points);

private:
    bool solveFrameByPnP(const int i, const std::vector<SFMFeature>& sfm_f, Eigen::Matrix3d& R_initial,
                         Eigen::Vector3d& P_initial);

    void triangulatePoint(const Eigen::Matrix<double, 3, 4>& Pose0, const Eigen::Matrix<double, 3, 4>& Pose1,
                          const Eigen::Vector2d& point0, const Eigen::Vector2d& point1, Eigen::Vector3d& point_3d);
    void triangulateTwoFrames(const int frame0, const Eigen::Matrix<double, 3, 4>& Pose0, const int frame1,
                              const Eigen::Matrix<double, 3, 4>& Pose1, std::vector<SFMFeature>& sfm_f);

    int feature_num_;
};
