#include <algorithm>
#include "System.h"

#include <pangolin/pangolin.h>

System::System(std::string sConfig_file_) : bStart_backend_(true)
{
    std::string sConfig_file = sConfig_file_ + "euroc_config.yaml";

    LOG(INFO) << "begin load system config file: " << sConfig_file;
    readParameters(sConfig_file);

    trackerData_[0].readIntrinsicParameter(sConfig_file);

    estimator_.setParameter();
    ofs_pose_.open("./output/pose_output.txt", std::fstream::app | std::fstream::out);
    if (!ofs_pose_.is_open())
    {
        LOG(ERROR) << "ofs_pose_ is not open";
    }

    LOG(INFO) << "System() initialization over" << std::endl << std::endl;
}

System::~System()
{
    bStart_backend_ = false;

    pangolin::QuitAll();

    while (!feature_buf_.empty())
    {
        feature_buf_.pop();
    }

    while (!imu_buf_.empty())
    {
        imu_buf_.pop();
    }

    estimator_.clearState();

    ofs_pose_.close();
}

void System::SetProcessOver()
{
    bStart_backend_ = false;
}

void System::PubImuData(double dStampSec, const Eigen::Vector3d& vGyr, const Eigen::Vector3d& vAcc)
{
    std::shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
    imu_msg->header              = dStampSec;
    imu_msg->linear_acceleration = vAcc;
    imu_msg->angular_velocity    = vGyr;

    if (dStampSec <= last_imu_t_)
    {
        LOG(ERROR) << "imu message in disorder!";
        return;
    }
    last_imu_t_ = dStampSec;
    // std::cout  << "1 PubImuData t: " << std::fixed << imu_msg->header
    //     << " acc: " << imu_msg->linear_acceleration.transpose()
    //     << " gyr: " << imu_msg->angular_velocity.transpose() << std::endl;
    std::lock_guard<std::mutex> lk(m_buf_);
    imu_buf_.push(imu_msg);
    // std::cout  << "1 PubImuData t: " << std::fixed << imu_msg->header
    //     << " imu_buf_ size:" << imu_buf_.size() << std::endl;
}

void System::PubImageData(double dStampSec, cv::Mat& img)
{
    if (!init_feature_)
    {
        LOG(ERROR) << "PubImageData skip the first detected feature, which doesn't contain optical flow speed";
        init_feature_ = 1;
        return;
    }

    if (first_image_flag_)
    {
        LOG(ERROR) << "PubImageData first_image_flag_";
        first_image_flag_ = false;
        first_image_time_ = dStampSec;
        last_image_time_  = dStampSec;
        return;
    }

    // detect unstable camera stream
    if (dStampSec - last_image_time_ > 1.0 || dStampSec < last_image_time_)
    {
        LOG(ERROR) << "PubImageData image discontinue! reset the feature tracker!";
        first_image_flag_ = true;
        last_image_time_  = 0;
        pub_count_        = 1;
        return;
    }

    last_image_time_ = dStampSec;
    // frequency control
    if (round(1.0 * pub_count_ / (dStampSec - first_image_time_)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count_ / (dStampSec - first_image_time_) - FREQ) < 0.01 * FREQ)
        {
            first_image_time_ = dStampSec;
            pub_count_        = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    TicToc t_r;
    // std::cout  << "3 PubImageData t : " << dStampSec << std::endl;
    trackerData_[0].readImage(img, dStampSec);

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        completed |= trackerData_[0].updateID(i);

        if (!completed)
            break;
    }
    if (PUB_THIS_FRAME)
    {
        pub_count_++;
        std::shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        std::vector<std::set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            const auto& un_pts       = trackerData_[i].cur_un_pts;
            const auto& cur_pts      = trackerData_[i].cur_pts;
            const auto& ids          = trackerData_[i].ids;
            const auto& pts_velocity = trackerData_[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData_[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Eigen::Vector3d(x, y, z));
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }

            LOG(INFO) << "PubImageData --- useful points size : " << feature_points->points.size()
                      << ", all points size : " << ids.size();

            // skip the first image; since no optical speed on frist image
            if (!init_pub_)
            {
                LOG(ERROR) << "PubImage init_pub_ skip the first image!";
                init_pub_ = 1;
            }
            else
            {
                std::lock_guard<std::mutex> lk(m_buf_);
                feature_buf_.push(feature_points);
                // std::cout  << "5 PubImage t : " << fixed << feature_points->header
                //     << " feature_buf_ size: " << feature_buf_.size() << std::endl;
            }
        }
    }

    cv::Mat show_img;
    cv::cvtColor(img, show_img, CV_GRAY2RGB);
    if (SHOW_TRACK)
    {
        for (unsigned int j = 0; j < trackerData_[0].cur_pts.size(); j++)
        {
            double len = std::min(1.0, 1.0 * trackerData_[0].track_cnt[j] / WINDOW_SIZE);
            // color : [B G R]
            cv::circle(show_img, trackerData_[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }

        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
    }
    // std::cout  << "5 PubImage" << std::endl;
}

std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true)
    {
        if (imu_buf_.empty() || feature_buf_.empty())
        {
            // cerr << "1 imu_buf_.empty() || feature_buf_.empty()" << std::endl;
            return measurements;
        }

        // imus     : -------
        // features :         *-----*
        if (!(imu_buf_.back()->header > feature_buf_.front()->header + estimator_.td_))
        {
            LOG(WARNING) << "wait for imu, only should happen at the beginning sum_of_wait_: " << sum_of_wait_;
            sum_of_wait_++;
            return measurements;
        }

        // imus     :     ---------
        // features : *-----*-----*
        if (!(imu_buf_.front()->header < feature_buf_.front()->header + estimator_.td_))
        {
            LOG(WARNING) << "throw img, only should happen at the beginning";
            feature_buf_.pop();
            continue;
        }

        ImgConstPtr img_msg = feature_buf_.front();
        feature_buf_.pop();

        // imus     : ---------
        // features :     *------*
        std::vector<ImuConstPtr> IMUs;
        while (imu_buf_.front()->header < img_msg->header + estimator_.td_)
        {
            IMUs.emplace_back(imu_buf_.front());
            imu_buf_.pop();
        }

        // imus     :    ------
        // features :     *------*
        // std::cout  << "1 getMeasurements IMUs size: " << IMUs.size() << std::endl;
        IMUs.emplace_back(imu_buf_.front());
        if (IMUs.empty())
        {
            LOG(WARNING) << "no imu between two image";
        }
        // std::cout  << "1 getMeasurements img t: " << fixed << img_msg->header
        //     << " imu begin: "<< IMUs.front()->header
        //     << " end: " << IMUs.back()->header
        //     << std::endl;
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// thread: visual-inertial odometry
void System::ProcessBackEnd()
{
    LOG(INFO) << "1 ProcessBackEnd start";
    while (bStart_backend_)
    {
        TicToc backend_timer;
        backend_timer.tic();
        std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> measurements;

        std::unique_lock<std::mutex> lk(m_buf_);
        measurements = getMeasurements();
        if (measurements.empty())
        {
            lk.unlock();
            usleep(5000);
            continue;
        }
        else if (measurements.size() > 1)
        {
            LOG(WARNING) << "1 getMeasurements size: " << measurements.size()
                         << " imu sizes: " << measurements[0].first.size()
                         << " feature_buf_ size: " << feature_buf_.size() << " imu_buf_ size: " << imu_buf_.size()
                         << std::endl;
        }
        lk.unlock();

        LOG(INFO) << "ProcessBackEnd --- success get measurements, begin process backend";
        for (const auto& measurement : measurements)
        {
            const std::vector<ImuConstPtr> imu_msgs = measurement.first;
            const ImgConstPtr              img_msg  = measurement.second;
            double                         img_t    = img_msg->header + estimator_.td_;
            LOG(INFO) << "imu msg size : " << imu_msgs.size() << ", img point number : " << img_msg->points.size()
                      << std::endl;
            LOG(INFO) << "imu msg front timestamp : " << std::fixed << imu_msgs.front()->header
                      << ", back timestamp : " << imu_msgs.back()->header << std::endl;
            LOG(INFO) << "img msg front timestamp : " << std::fixed << img_msg->header << std::endl;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (const auto& imu_msg : imu_msgs)
            {
                double t = imu_msg->header;
                if (t <= img_t)
                {
                    if (current_time_ < 0)
                        current_time_ = t;
                    double dt = t - current_time_;
                    assert(dt >= 0);
                    current_time_ = t;
                    dx            = imu_msg->linear_acceleration.x();
                    dy            = imu_msg->linear_acceleration.y();
                    dz            = imu_msg->linear_acceleration.z();
                    rx            = imu_msg->angular_velocity.x();
                    ry            = imu_msg->angular_velocity.y();
                    rz            = imu_msg->angular_velocity.z();
                    estimator_.processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                    // printf("1 BackEnd imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    double dt_1   = img_t - current_time_;
                    double dt_2   = t - img_t;
                    current_time_ = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx        = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy        = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz        = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx        = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry        = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz        = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator_.processIMU(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                    LOG(WARNING) << "interpolation, imu timestamp : " << std::fixed << t
                                 << ", img timestamp : " << img_t << std::endl;
                }
            }

            // TicToc t_s;
            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int    v          = img_msg->id_of_point[i] + 0.5;
                int    feature_id = v / NUM_OF_CAM;
                int    camera_id  = v % NUM_OF_CAM;
                double x          = img_msg->points[i].x();
                double y          = img_msg->points[i].y();
                double z          = img_msg->points[i].z();
                double p_u        = img_msg->u_of_point[i];
                double p_v        = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            TicToc t_processImage;
            estimator_.processImage(image, img_msg->header);

            if (estimator_.solver_flag_ == Estimator::SolverFlag::NON_LINEAR)
            {
                Eigen::Vector3d    p_wi;
                Eigen::Quaterniond q_wi;
                q_wi = Eigen::Quaterniond(estimator_.Rs_[WINDOW_SIZE]);
                p_wi = estimator_.Ps_[WINDOW_SIZE];
                vPath_to_draw_.push_back(p_wi);
                double dStamp = estimator_.Headers_[WINDOW_SIZE];
                LOG(INFO) << "1 BackEnd processImage dt: " << std::fixed << t_processImage.toc()
                          << " ms, stamp: " << dStamp << ", p_wi: " << p_wi.transpose();
                ofs_pose_ << std::fixed << dStamp << " " << p_wi.transpose() << " " << q_wi.coeffs().transpose()
                          << std::endl;
            }
        }

        LOG(INFO) << "ProcessBackEnd --- over process backend, cost : " << backend_timer.toc() << " ms.";
    }
}

void System::Draw()
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam_ = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 100000),
                                         pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0));

    d_cam_ = pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                 .SetHandler(new pangolin::Handler3D(s_cam_));

    while (bStart_backend_)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0, 1.0, 1.0, 1.0);

        d_cam_.Activate(s_cam_);
        glLineWidth(2);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);

        // draw poses
        glColor3f(0, 0, 0);
        glPointSize(3);
        glBegin(GL_POINTS);
        // TODO: @chengchangxi, this shared memory should be locked, this thread has data race with backend thread
        int nPath_size = vPath_to_draw_.size();
        for (int i = 0; i < nPath_size; ++i)
        {
            glVertex3f(vPath_to_draw_[i].x(), vPath_to_draw_[i].y(), vPath_to_draw_[i].z());
        }
        glEnd();

        // points
        if (estimator_.solver_flag_ == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(3);
            glBegin(GL_POINTS);
            for (int i = 0; i < WINDOW_SIZE + 1; ++i)
            {
                Eigen::Vector3d p_wi = estimator_.Ps_[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
    }
}
