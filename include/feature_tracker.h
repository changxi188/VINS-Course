#pragma once

#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "utility/tic_toc.h"

using namespace camodocal;

bool inBorder(const cv::Point2f& pt);

void reduceVector(std::vector<cv::Point2f>& v, std::vector<uchar> status);
void reduceVector(std::vector<int>& v, std::vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();

    void readImage(const cv::Mat& _img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const std::string& calib_file);

    void showUndistortion(const std::string& name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat                    mask;
    cv::Mat                    fisheye_mask;
    cv::Mat                    prev_img, cur_img, forw_img;
    std::vector<cv::Point2f>   n_pts;
    std::vector<cv::Point2f>   prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f>   prev_un_pts, cur_un_pts;
    std::vector<cv::Point2f>   pts_velocity;
    std::vector<int>           ids;
    std::vector<int>           track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map;
    std::map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr       m_camera;
    double                     cur_time;
    double                     prev_time;

    static int n_id;
};
