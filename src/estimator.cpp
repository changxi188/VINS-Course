#include "estimator.h"

#include "backend/edge_imu.h"
#include "backend/edge_reprojection.h"
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"

#include <fstream>
#include <ostream>

using namespace myslam;

Estimator::Estimator() : f_manager_{Rs_}
{
    // ROS_INFO("init begins");

    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        pre_integrations_[i] = nullptr;
    }
    for (auto& it : all_image_frame_)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration_ = nullptr;

    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = TIC[i];
        ric_[i] = RIC[i];
        // cout << "1 Estimator::setParameter tic_: " << tic_[i].transpose()
        //     << " ric_: " << ric_[i] << endl;
    }
    LOG(INFO) << "Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH;
    f_manager_.setRic(ric_);
    project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
    td_                = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs_[i].setIdentity();
        Ps_[i].setZero();
        Vs_[i].setZero();
        Bas_[i].setZero();
        Bgs_[i].setZero();
        dt_buf_[i].clear();
        linear_acceleration_buf_[i].clear();
        angular_velocity_buf_[i].clear();

        if (pre_integrations_[i] != nullptr)
            delete pre_integrations_[i];
        pre_integrations_[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = Eigen::Vector3d::Zero();
        ric_[i] = Eigen::Matrix3d::Identity();
    }

    for (auto& it : all_image_frame_)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag_ = INITIAL;
    first_imu_ = true, sum_of_back_ = 0;
    sum_of_front_      = 0;
    frame_count_       = 0;
    solver_flag_       = INITIAL;
    initial_timestamp_ = 0;
    all_image_frame_.clear();
    td_ = TD;

    if (tmp_pre_integration_ != nullptr)
        delete tmp_pre_integration_;

    tmp_pre_integration_ = nullptr;

    last_marginalization_parameter_blocks_.clear();

    f_manager_.clearState();

    failure_occur_       = 0;
    relocalization_info_ = 0;

    drift_correct_r_ = Eigen::Matrix3d::Identity();
    drift_correct_t_ = Eigen::Vector3d::Zero();
}

void Estimator::processIMU(double dt, const Eigen::Vector3d& linear_acceleration,
                           const Eigen::Vector3d& angular_velocity)
{
    if (first_imu_)
    {
        first_imu_ = false;
        acc_0_     = linear_acceleration;
        gyr_0_     = angular_velocity;
    }

    if (!pre_integrations_[frame_count_])
    {
        pre_integrations_[frame_count_] = new IntegrationBase{acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_]};
    }

    if (frame_count_ != 0)
    {
        pre_integrations_[frame_count_]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag_ != NON_LINEAR)
        tmp_pre_integration_->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf_[frame_count_].push_back(dt);
        linear_acceleration_buf_[frame_count_].push_back(linear_acceleration);
        angular_velocity_buf_[frame_count_].push_back(angular_velocity);

        int             j        = frame_count_;
        Eigen::Vector3d un_acc_0 = Rs_[j] * (acc_0_ - Bas_[j]) - g_;
        Eigen::Vector3d un_gyr   = 0.5 * (gyr_0_ + angular_velocity) - Bgs_[j];
        Rs_[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs_[j] * (linear_acceleration - Bas_[j]) - g_;
        Eigen::Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
        Ps_[j] += dt * Vs_[j] + 0.5 * dt * dt * un_acc;
        Vs_[j] += dt * un_acc;
    }
    acc_0_ = linear_acceleration;
    gyr_0_ = angular_velocity;
}

void Estimator::processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                             double                                                                         header)
{
    // ROS_DEBUG("new image coming ------------------------------------------");
    //  cout << "Adding feature points: " << image.size()<<endl;
    if (f_manager_.addFeatureCheckParallax(frame_count_, image, td_))
        marginalization_flag_ = MARGIN_OLD;
    else
        marginalization_flag_ = MARGIN_SECOND_NEW;

    // ROS_DEBUG("this frame is--------------------%s", marginalization_flag_ ? "reject" : "accept");
    // ROS_DEBUG("%s", marginalization_flag_ ? "Non-keyframe" : "Keyframe");
    // ROS_DEBUG("Solving %d", frame_count_);
    //  cout << "number of feature: " << f_manager_.getFeatureCount()<<endl;
    Headers_[frame_count_] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration_;
    all_image_frame_.insert(std::make_pair(header, imageframe));
    tmp_pre_integration_ = new IntegrationBase{acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_]};

    if (ESTIMATE_EXTRINSIC == 2)
    {
        LOG(WARNING) << "calibrating extrinsic param, rotation movement is needed";
        if (frame_count_ != 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres =
                f_manager_.getCorresponding(frame_count_ - 1, frame_count_);
            Eigen::Matrix3d calib_ric;
            if (initial_ex_rotation_.CalibrationExRotation(corres, pre_integrations_[frame_count_]->delta_q_,
                                                           calib_ric))
            {
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                //    << calib_ric);
                ric_[0]            = calib_ric;
                RIC[0]             = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag_ == INITIAL)
    {
        if (frame_count_ == WINDOW_SIZE)
        {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp_) > 0.1)
            {
                // cout << "1 initialStructure" << endl;
                result             = initialStructure();
                initial_timestamp_ = header;
            }
            if (result)
            {
                solver_flag_ = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager_.removeFailures();
                std::cout << "Initialization finish!" << std::endl;
                last_R_  = Rs_[WINDOW_SIZE];
                last_P_  = Ps_[WINDOW_SIZE];
                last_R0_ = Rs_[0];
                last_P0_ = Ps_[0];
            }
            else
                slideWindow();
        }
        else
            frame_count_++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        // ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            // ROS_WARN("failure detection!");
            failure_occur_ = 1;
            clearState();
            setParameter();
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager_.removeFailures();
        // ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        //  prepare output of VINS
        key_poses_.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses_.push_back(Ps_[i]);

        last_R_  = Rs_[WINDOW_SIZE];
        last_P_  = Ps_[WINDOW_SIZE];
        last_R0_ = Rs_[0];
        last_P0_ = Ps_[0];
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d                        sum_g;
        for (frame_it = all_image_frame_.begin(), frame_it++; frame_it != all_image_frame_.end(); frame_it++)
        {
            double          dt    = frame_it->second.pre_integration->sum_dt_;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v_ / dt;
            sum_g += tmp_g;
        }
        Eigen::Vector3d aver_g;
        aver_g     = sum_g * 1.0 / ((int)all_image_frame_.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame_.begin(), frame_it++; frame_it != all_image_frame_.end(); frame_it++)
        {
            double          dt    = frame_it->second.pre_integration->sum_dt_;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v_ / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame_.size() - 1));
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            // ROS_INFO("IMU excitation not enouth!");
            // return false;
        }
    }
    // global sfm
    Eigen::Quaterniond             Q[frame_count_ + 1];
    Eigen::Vector3d                T[frame_count_ + 1];
    std::map<int, Eigen::Vector3d> sfm_tracked_points;
    std::vector<SFMFeature>        sfm_f;
    for (auto& it_per_id : f_manager_.feature_)
    {
        int        imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id    = it_per_id.feature_id;
        for (auto& it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Eigen::Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(std::make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int             l;
    if (!relativePose(relative_R, relative_T, l))
    {
        std::cout << "Not enough features or parallax; Move device around" << std::endl;
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count_ + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        std::cout << "global SFM failed!" << std::endl;
        marginalization_flag_ = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    std::map<double, ImageFrame>::iterator   frame_it;
    std::map<int, Eigen::Vector3d>::iterator it;
    frame_it = all_image_frame_.begin();
    for (int i = 0; frame_it != all_image_frame_.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers_[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R            = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T            = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers_[i])
        {
            i++;
        }
        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto& id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto& i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Eigen::Vector3d world_pts = it->second;
                    cv::Point3f     pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Eigen::Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f     pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            std::cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << std::endl;
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            std::cout << " solve pnp fail!" << std::endl;
            return false;
        }
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        Eigen::MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp              = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        std::cout << "misalign visual structure with IMU" << std::endl;
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc          t_g;
    Eigen::VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame_, Bgs_, g_, x);
    if (!result)
    {
        // ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count_; i++)
    {
        Eigen::Matrix3d Ri                         = all_image_frame_[Headers_[i]].R;
        Eigen::Vector3d Pi                         = all_image_frame_[Headers_[i]].T;
        Ps_[i]                                     = Pi;
        Rs_[i]                                     = Ri;
        all_image_frame_[Headers_[i]].is_key_frame = true;
    }

    Eigen::VectorXd dep = f_manager_.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager_.clearDepth(dep);

    // triangulat on cam pose , no tic_
    Eigen::Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric_[0] = RIC[0];
    f_manager_.setRic(ric_);
    f_manager_.triangulate(Ps_, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations_[i]->repropagate(Eigen::Vector3d::Zero(), Bgs_[i]);
    }
    for (int i = frame_count_; i >= 0; i--)
        Ps_[i] = s * Ps_[i] - Rs_[i] * TIC[0] - (s * Ps_[0] - Rs_[0] * TIC[0]);
    int                                    kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame_.begin(); frame_i != all_image_frame_.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs_[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto& it_per_id : f_manager_.feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Eigen::Matrix3d R0  = Utility::g2R(g_);
    double          yaw = Utility::R2ypr(R0 * Rs_[0]).x();
    R0                  = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g_                  = R0 * g_;
    // Matrix3d rot_diff = R0 * Rs_[0].transpose();
    Eigen::Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count_; i++)
    {
        Ps_[i] = rot_diff * Ps_[i];
        Rs_[i] = rot_diff * Rs_[i];
        Vs_[i] = rot_diff * Vs_[i];
    }
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs_[0]).transpose());

    return true;
}

bool Estimator::relativePose(Eigen::Matrix3d& relative_R, Eigen::Vector3d& relative_T, int& l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        corres = f_manager_.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double          parallax = (pts_0 - pts_1).norm();
                sum_parallax             = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator_.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                // ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                // average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count_ < WINDOW_SIZE)
        return;
    if (solver_flag_ == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager_.triangulate(Ps_, tic_, ric_);
        // cout << "triangulation costs : " << t_tri.toc() << endl;
        backendOptimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose_[i][0] = Ps_[i].x();
        para_Pose_[i][1] = Ps_[i].y();
        para_Pose_[i][2] = Ps_[i].z();
        Eigen::Quaterniond q{Rs_[i]};
        para_Pose_[i][3] = q.x();
        para_Pose_[i][4] = q.y();
        para_Pose_[i][5] = q.z();
        para_Pose_[i][6] = q.w();

        para_SpeedBias_[i][0] = Vs_[i].x();
        para_SpeedBias_[i][1] = Vs_[i].y();
        para_SpeedBias_[i][2] = Vs_[i].z();

        para_SpeedBias_[i][3] = Bas_[i].x();
        para_SpeedBias_[i][4] = Bas_[i].y();
        para_SpeedBias_[i][5] = Bas_[i].z();

        para_SpeedBias_[i][6] = Bgs_[i].x();
        para_SpeedBias_[i][7] = Bgs_[i].y();
        para_SpeedBias_[i][8] = Bgs_[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose_[i][0] = tic_[i].x();
        para_Ex_Pose_[i][1] = tic_[i].y();
        para_Ex_Pose_[i][2] = tic_[i].z();
        Eigen::Quaterniond q{ric_[i]};
        para_Ex_Pose_[i][3] = q.x();
        para_Ex_Pose_[i][4] = q.y();
        para_Ex_Pose_[i][5] = q.z();
        para_Ex_Pose_[i][6] = q.w();
    }

    Eigen::VectorXd dep = f_manager_.getDepthVector();
    for (int i = 0; i < f_manager_.getFeatureCount(); i++)
        para_Feature_[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td_[0][0] = td_;
}

void Estimator::double2vector()
{
    Eigen::Vector3d origin_R0 = Utility::R2ypr(Rs_[0]);
    Eigen::Vector3d origin_P0 = Ps_[0];

    if (failure_occur_)
    {
        origin_R0      = Utility::R2ypr(last_R0_);
        origin_P0      = last_P0_;
        failure_occur_ = 0;
    }
    Eigen::Vector3d origin_R00 = Utility::R2ypr(
        Eigen::Quaterniond(para_Pose_[0][6], para_Pose_[0][3], para_Pose_[0][4], para_Pose_[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Eigen::Matrix3d rot_diff = Utility::ypr2R(Eigen::Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        // ROS_DEBUG("euler singular point!");
        rot_diff = Rs_[0] * Eigen::Quaterniond(para_Pose_[0][6], para_Pose_[0][3], para_Pose_[0][4], para_Pose_[0][5])
                                .toRotationMatrix()
                                .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs_[i] = rot_diff * Eigen::Quaterniond(para_Pose_[i][6], para_Pose_[i][3], para_Pose_[i][4], para_Pose_[i][5])
                                .normalized()
                                .toRotationMatrix();

        Ps_[i] = rot_diff * Eigen::Vector3d(para_Pose_[i][0] - para_Pose_[0][0], para_Pose_[i][1] - para_Pose_[0][1],
                                            para_Pose_[i][2] - para_Pose_[0][2]) +
                 origin_P0;

        Vs_[i] = rot_diff * Eigen::Vector3d(para_SpeedBias_[i][0], para_SpeedBias_[i][1], para_SpeedBias_[i][2]);

        Bas_[i] = Eigen::Vector3d(para_SpeedBias_[i][3], para_SpeedBias_[i][4], para_SpeedBias_[i][5]);

        Bgs_[i] = Eigen::Vector3d(para_SpeedBias_[i][6], para_SpeedBias_[i][7], para_SpeedBias_[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = Eigen::Vector3d(para_Ex_Pose_[i][0], para_Ex_Pose_[i][1], para_Ex_Pose_[i][2]);
        ric_[i] = Eigen::Quaterniond(para_Ex_Pose_[i][6], para_Ex_Pose_[i][3], para_Ex_Pose_[i][4], para_Ex_Pose_[i][5])
                      .toRotationMatrix();
    }

    Eigen::VectorXd dep = f_manager_.getDepthVector();
    for (int i = 0; i < f_manager_.getFeatureCount(); i++)
        dep(i) = para_Feature_[i][0];
    f_manager_.setDepth(dep);
    if (ESTIMATE_TD)
        td_ = para_Td_[0][0];

    // relative info between two loop frame
    if (relocalization_info_)
    {
        Eigen::Matrix3d relo_r;
        Eigen::Vector3d relo_t;
        relo_r = rot_diff * Eigen::Quaterniond(relo_Pose_[6], relo_Pose_[3], relo_Pose_[4], relo_Pose_[5])
                                .normalized()
                                .toRotationMatrix();
        relo_t = rot_diff * Eigen::Vector3d(relo_Pose_[0] - para_Pose_[0][0], relo_Pose_[1] - para_Pose_[0][1],
                                            relo_Pose_[2] - para_Pose_[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r_).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r_  = Utility::ypr2R(Eigen::Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t_  = prev_relo_t_ - drift_correct_r_ * relo_t;
        relo_relative_t_  = relo_r.transpose() * (Ps_[relo_frame_local_index_] - relo_t);
        relo_relative_q_  = relo_r.transpose() * Rs_[relo_frame_local_index_];
        relo_relative_yaw_ =
            Utility::normalizeAngle(Utility::R2ypr(Rs_[relo_frame_local_index_]).x() - Utility::R2ypr(relo_r).x());
        // cout << "vins relo " << endl;
        // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info_ = 0;
    }
}

bool Estimator::failureDetection()
{
    if (f_manager_.last_track_num_ < 2)
    {
        // ROS_INFO(" little feature %d", f_manager_.last_track_num);
        // return true;
    }
    if (Bas_[WINDOW_SIZE].norm() > 2.5)
    {
        // ROS_INFO(" big IMU acc bias estimation %f", Bas_[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs_[WINDOW_SIZE].norm() > 1.0)
    {
        // ROS_INFO(" big IMU gyr bias estimation %f", Bgs_[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic_(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic_(0) > 1);
        return true;
    }
    */
    Eigen::Vector3d tmp_P = Ps_[WINDOW_SIZE];
    if ((tmp_P - last_P_).norm() > 5)
    {
        // ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P_.z()) > 1)
    {
        // ROS_INFO(" big z translation");
        return true;
    }
    Eigen::Matrix3d    tmp_R   = Rs_[WINDOW_SIZE];
    Eigen::Matrix3d    delta_R = tmp_R.transpose() * last_R_;
    Eigen::Quaterniond delta_Q(delta_R);
    double             delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        // ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

void Estimator::MargOldFrame()
{
    backend::LossFunction* lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);

    // step1. 构建 problem
    backend::Problem                                       problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    std::vector<std::shared_ptr<backend::VertexPose>>      vertexCams_vec;
    std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int                                                    pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose_[0][0], para_Ex_Pose_[0][1], para_Ex_Pose_[0][2], para_Ex_Pose_[0][3], para_Ex_Pose_[0][4],
            para_Ex_Pose_[0][5], para_Ex_Pose_[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd                      pose(7);
        pose << para_Pose_[i][0], para_Pose_[i][1], para_Pose_[i][2], para_Pose_[i][3], para_Pose_[i][4],
            para_Pose_[i][5], para_Pose_[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd                           vb(9);
        vb << para_SpeedBias_[i][0], para_SpeedBias_[i][1], para_SpeedBias_[i][2], para_SpeedBias_[i][3],
            para_SpeedBias_[i][4], para_SpeedBias_[i][5], para_SpeedBias_[i][6], para_SpeedBias_[i][7],
            para_SpeedBias_[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU
    {
        if (pre_integrations_[1]->sum_dt_ < 10.0)
        {
            std::shared_ptr<backend::EdgeImu>             imuEdge(new backend::EdgeImu(pre_integrations_[1]));
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexVB_vec[0]);
            edge_vertex.push_back(vertexCams_vec[1]);
            edge_vertex.push_back(vertexVB_vec[1]);
            imuEdge->SetVertex(edge_vertex);
            problem.AddEdge(imuEdge);
        }
    }

    // Visual Factor
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto& it_per_id : f_manager_.feature_)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;

            Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX                                         inv_d(1);
            inv_d << para_Feature_[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);

            // 遍历所有的观测
            for (auto& it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Eigen::Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection>    edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_);  // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15);  // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
            problem.SetHessianPrior(Hprior_);  // 告诉这个 problem
            problem.SetbPrior(bprior_);
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    marg_vertex.push_back(vertexCams_vec[0]);
    marg_vertex.push_back(vertexVB_vec[0]);
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_     = problem.GetHessianPrior();
    bprior_     = problem.GetbPrior();
    errprior_   = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}
void Estimator::MargNewFrame()
{
    // step1. 构建 problem
    backend::Problem                                       problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    std::vector<std::shared_ptr<backend::VertexPose>>      vertexCams_vec;
    std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    //    vector<backend::Point3d> points;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose_[0][0], para_Ex_Pose_[0][1], para_Ex_Pose_[0][2], para_Ex_Pose_[0][3], para_Ex_Pose_[0][4],
            para_Ex_Pose_[0][5], para_Ex_Pose_[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd                      pose(7);
        pose << para_Pose_[i][0], para_Pose_[i][1], para_Pose_[i][2], para_Pose_[i][3], para_Pose_[i][4],
            para_Pose_[i][5], para_Pose_[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd                           vb(9);
        vb << para_SpeedBias_[i][0], para_SpeedBias_[i][1], para_SpeedBias_[i][2], para_SpeedBias_[i][3],
            para_SpeedBias_[i][4], para_SpeedBias_[i][5], para_SpeedBias_[i][6], para_SpeedBias_[i][7],
            para_SpeedBias_[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_);  // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);

            problem.ExtendHessiansPriorSize(15);  // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    // 把窗口倒数第二个帧 marg 掉
    marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);
    marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_     = problem.GetHessianPrior();
    bprior_     = problem.GetbPrior();
    errprior_   = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}

void Estimator::problemSolve()
{
    backend::LossFunction* lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // step1. 构建 problem
    backend::Problem                                       problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    std::vector<std::shared_ptr<backend::VertexPose>>      vertexCams_vec;
    std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int                                                    pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose_[0][0], para_Ex_Pose_[0][1], para_Ex_Pose_[0][2], para_Ex_Pose_[0][3], para_Ex_Pose_[0][4],
            para_Ex_Pose_[0][5], para_Ex_Pose_[0][6];
        vertexExt->SetParameters(pose);

        if (!ESTIMATE_EXTRINSIC)
        {
            // ROS_DEBUG("fix extinsic param");
            //  TODO:: set Hessian prior to zero
            vertexExt->SetFixed();
        }
        else
        {
            // ROS_DEBUG("estimate extinsic param");
        }
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd                      pose(7);
        pose << para_Pose_[i][0], para_Pose_[i][1], para_Pose_[i][2], para_Pose_[i][3], para_Pose_[i][4],
            para_Pose_[i][5], para_Pose_[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd                           vb(9);
        vb << para_SpeedBias_[i][0], para_SpeedBias_[i][1], para_SpeedBias_[i][2], para_SpeedBias_[i][3],
            para_SpeedBias_[i][4], para_SpeedBias_[i][5], para_SpeedBias_[i][6], para_SpeedBias_[i][7],
            para_SpeedBias_[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations_[j]->sum_dt_ > 10.0)
            continue;

        std::shared_ptr<backend::EdgeImu>             imuEdge(new backend::EdgeImu(pre_integrations_[j]));
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams_vec[i]);
        edge_vertex.push_back(vertexVB_vec[i]);
        edge_vertex.push_back(vertexCams_vec[j]);
        edge_vertex.push_back(vertexVB_vec[j]);
        imuEdge->SetVertex(edge_vertex);
        problem.AddEdge(imuEdge);
    }

    // Visual Factor
    std::vector<std::shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto& it_per_id : f_manager_.feature_)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int             imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX                                         inv_d(1);
            inv_d << para_Feature_[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);
            vertexPt_vec.push_back(verterxPoint);

            // 遍历所有的观测
            for (auto& it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Eigen::Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection>    edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

            problem.SetHessianPrior(Hprior_);  // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15);  // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
    }

    problem.Solve(10);

    // update bprior_,  Hprior_ do not need update
    if (Hprior_.rows() > 0)
    {
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << bprior_.norm() << std::endl;
        std::cout << "                     " << errprior_.norm() << std::endl;
        bprior_   = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        std::cout << "             after: " << bprior_.norm() << std::endl;
        std::cout << "                    " << errprior_.norm() << std::endl;
    }

    // update parameter
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        VecX p = vertexCams_vec[i]->Parameters();
        for (int j = 0; j < 7; ++j)
        {
            para_Pose_[i][j] = p[j];
        }

        VecX vb = vertexVB_vec[i]->Parameters();
        for (int j = 0; j < 9; ++j)
        {
            para_SpeedBias_[i][j] = vb[j];
        }
    }

    // 遍历每一个特征
    for (size_t i = 0; i < vertexPt_vec.size(); ++i)
    {
        VecX f              = vertexPt_vec[i]->Parameters();
        para_Feature_[i][0] = f[0];
    }
}

void Estimator::backendOptimization()
{
    TicToc t_solver;
    // 借助 vins 框架，维护变量
    vector2double();
    // 构建求解器
    problemSolve();
    // 优化后的变量处理下自由度
    double2vector();
    // ROS_INFO("whole time for solver: %f", t_solver.toc());

    // 维护 marg
    TicToc t_whole_marginalization;
    if (marginalization_flag_ == MARGIN_OLD)
    {
        vector2double();

        MargOldFrame();

        std::unordered_map<long, double*> addr_shift;  // prior 中对应的保留下来的参数地址
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose_[i])]      = para_Pose_[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];
        }
    }
    else
    {
        if (Hprior_.rows() > 0)
        {
            vector2double();

            MargNewFrame();

            std::unordered_map<long, double*> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose_[i])]      = para_Pose_[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose_[i])]      = para_Pose_[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];
            }
        }
    }
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag_ == MARGIN_OLD)
    {
        double t_0 = Headers_[0];
        back_R0_   = Rs_[0];
        back_P0_   = Ps_[0];
        // if (frame_count_ == WINDOW_SIZE)
        // {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            Rs_[i].swap(Rs_[i + 1]);

            std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

            dt_buf_[i].swap(dt_buf_[i + 1]);
            linear_acceleration_buf_[i].swap(linear_acceleration_buf_[i + 1]);
            angular_velocity_buf_[i].swap(angular_velocity_buf_[i + 1]);

            Headers_[i] = Headers_[i + 1];
            Ps_[i].swap(Ps_[i + 1]);
            Vs_[i].swap(Vs_[i + 1]);
            Bas_[i].swap(Bas_[i + 1]);
            Bgs_[i].swap(Bgs_[i + 1]);
        }
        Headers_[WINDOW_SIZE] = Headers_[WINDOW_SIZE - 1];
        Ps_[WINDOW_SIZE]      = Ps_[WINDOW_SIZE - 1];
        Vs_[WINDOW_SIZE]      = Vs_[WINDOW_SIZE - 1];
        Rs_[WINDOW_SIZE]      = Rs_[WINDOW_SIZE - 1];
        Bas_[WINDOW_SIZE]     = Bas_[WINDOW_SIZE - 1];
        Bgs_[WINDOW_SIZE]     = Bgs_[WINDOW_SIZE - 1];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] = new IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();

        // if (true || solver_flag_ == INITIAL)
        // {
        std::map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame_.find(t_0);
        delete it_0->second.pre_integration;
        it_0->second.pre_integration = nullptr;

        for (std::map<double, ImageFrame>::iterator it = all_image_frame_.begin(); it != it_0; ++it)
        {
            if (it->second.pre_integration)
                delete it->second.pre_integration;
            it->second.pre_integration = NULL;
        }

        all_image_frame_.erase(all_image_frame_.begin(), it_0);
        all_image_frame_.erase(t_0);
        slideWindowOld();
        // }
        // }
    }
    else
    {
        // if (frame_count_ == WINDOW_SIZE)
        // {
        for (unsigned int i = 0; i < dt_buf_[frame_count_].size(); i++)
        {
            double          tmp_dt                  = dt_buf_[frame_count_][i];
            Eigen::Vector3d tmp_linear_acceleration = linear_acceleration_buf_[frame_count_][i];
            Eigen::Vector3d tmp_angular_velocity    = angular_velocity_buf_[frame_count_][i];

            pre_integrations_[frame_count_ - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

            dt_buf_[frame_count_ - 1].push_back(tmp_dt);
            linear_acceleration_buf_[frame_count_ - 1].push_back(tmp_linear_acceleration);
            angular_velocity_buf_[frame_count_ - 1].push_back(tmp_angular_velocity);
        }

        Headers_[frame_count_ - 1] = Headers_[frame_count_];
        Ps_[frame_count_ - 1]      = Ps_[frame_count_];
        Vs_[frame_count_ - 1]      = Vs_[frame_count_];
        Rs_[frame_count_ - 1]      = Rs_[frame_count_];
        Bas_[frame_count_ - 1]     = Bas_[frame_count_];
        Bgs_[frame_count_ - 1]     = Bgs_[frame_count_];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] = new IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();

        slideWindowNew();
        // }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front_++;
    f_manager_.removeFront(frame_count_);
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back_++;

    bool shift_depth = solver_flag_ == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = back_R0_ * ric_[0];
        R1 = Rs_[0] * ric_[0];
        P0 = back_P0_ + back_R0_ * tic_[0];
        P1 = Ps_[0] + Rs_[0] * tic_[0];
        f_manager_.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager_.removeBack();
}
