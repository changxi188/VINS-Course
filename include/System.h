#pragma once

#include <stdio.h>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include <pangolin/pangolin.h>

#include "estimator.h"
#include "feature_tracker.h"
#include "parameters.h"

// imu for vio
struct IMU_MSG
{
    double          header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

// image for vio
struct IMG_MSG
{
    double                       header;
    std::vector<Eigen::Vector3d> points;
    std::vector<int>             id_of_point;
    std::vector<float>           u_of_point;
    std::vector<float>           v_of_point;
    std::vector<float>           velocity_x_of_point;
    std::vector<float>           velocity_y_of_point;
};
typedef std::shared_ptr<IMG_MSG const> ImgConstPtr;

class System
{
public:
    System(std::string sConfig_files);

    ~System();

    void PubImageData(double dStampSec, cv::Mat& img);

    void PubImuData(double dStampSec, const Eigen::Vector3d& vGyr, const Eigen::Vector3d& vAcc);

    // thread: visual-inertial odometry
    void ProcessBackEnd();

    void Draw();

    void SetProcessOver();

private:
    // feature tracker
    std::vector<uchar> r_status_;
    std::vector<float> r_err_;

    FeatureTracker trackerData_[NUM_OF_CAM];
    double         first_image_time_;
    int            pub_count_        = 1;
    bool           first_image_flag_ = true;
    double         last_image_time_  = 0;
    bool           init_pub_         = 0;

    // estimator
    Estimator estimator_;

    double                  current_time_ = -1;
    std::queue<ImuConstPtr> imu_buf_;
    std::queue<ImgConstPtr> feature_buf_;
    // std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait_ = 0;

    // visualizer
    pangolin::OpenGlRenderState s_cam_;
    pangolin::View              d_cam_;

    std::mutex m_buf_;

    double                                                        latest_time_;
    Eigen::Vector3d                                               tmp_P_;
    Eigen::Quaterniond                                            tmp_Q_;
    Eigen::Vector3d                                               tmp_V_;
    Eigen::Vector3d                                               tmp_Ba_;
    Eigen::Vector3d                                               tmp_Bg_;
    Eigen::Vector3d                                               acc_0_;
    Eigen::Vector3d                                               gyr_0_;
    bool                                                          init_feature_ = 0;
    bool                                                          init_imu_     = 1;
    double                                                        last_imu_t_   = 0;
    std::ofstream                                                 ofs_pose_;
    std::vector<Eigen::Vector3d>                                  vPath_to_draw_;
    std::atomic<bool>                                             bStart_backend_;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
};
