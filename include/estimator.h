#pragma once

#include "feature_manager.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_sfm.h"
#include "initial/solve_5pts.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

#include "factor/integration_base.h"

#include "backend/problem.h"

#include <opencv2/core/eigen.hpp>
#include <queue>
#include <unordered_map>

class Estimator
{
public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Eigen::Vector3d& linear_acceleration, const Eigen::Vector3d& angular_velocity);

    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                      double                                                                         header);
    void setReloFrame(double _frame_stamp, int _frame_index, std::vector<Eigen::Vector3d>& _match_points,
                      Eigen::Vector3d _relo_t, Eigen::Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Eigen::Matrix3d& relative_R, Eigen::Vector3d& relative_T, int& l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void backendOptimization();

    void problemSolve();
    void MargOldFrame();
    void MargNewFrame();

    void vector2double();
    void double2vector();
    bool failureDetection();

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD        = 0,
        MARGIN_SECOND_NEW = 1
    };

    FeatureManager    f_manager_;
    MotionEstimator   m_estimator_;
    InitialEXRotation initial_ex_rotation_;

    //////////////// OUR SOLVER ///////////////////
    MatXX Hprior_;
    VecX  bprior_;
    VecX  errprior_;
    MatXX Jprior_inv_;

    Eigen::Matrix2d project_sqrt_info_;
    //////////////// OUR SOLVER //////////////////
    SolverFlag          solver_flag_;
    MarginalizationFlag marginalization_flag_;
    Eigen::Vector3d     g_;
    Eigen::MatrixXd     Ap_[2], backup_A_;
    Eigen::VectorXd     bp_[2], backup_b_;

    Eigen::Matrix3d ric_[NUM_OF_CAM];
    Eigen::Vector3d tic_[NUM_OF_CAM];

    Eigen::Vector3d Ps_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs_[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d Rs_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs_[(WINDOW_SIZE + 1)];
    double          Headers_[(WINDOW_SIZE + 1)];

    double          td_;
    Eigen::Matrix3d back_R0_, last_R_, last_R0_;
    Eigen::Vector3d back_P0_, last_P_, last_P0_;

    IntegrationBase* pre_integrations_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d  acc_0_, gyr_0_;

    std::vector<double>          dt_buf_[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> linear_acceleration_buf_[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> angular_velocity_buf_[(WINDOW_SIZE + 1)];

    int frame_count_;
    int sum_of_outlier_, sum_of_back_, sum_of_front_, sum_of_invalid_;

    bool first_imu_;
    bool is_valid_, is_key_;
    bool failure_occur_;

    std::vector<Eigen::Vector3d> point_cloud_;
    std::vector<Eigen::Vector3d> margin_cloud_;
    std::vector<Eigen::Vector3d> key_poses_;
    double                       initial_timestamp_;

    double para_Pose_[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias_[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature_[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose_[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose_[SIZE_POSE];
    double para_Td_[1][1];
    double para_Tr_[1][1];

    int loop_window_index_;

    // MarginalizationInfo *last_marginalization_info;
    std::vector<double*> last_marginalization_parameter_blocks_;

    // timestamp to ImageFrame
    std::map<double, ImageFrame> all_image_frame_;
    IntegrationBase*             tmp_pre_integration_;

    // relocalization variable
    bool                         relocalization_info_;
    double                       relo_frame_stamp_;
    double                       relo_frame_index_;
    int                          relo_frame_local_index_;
    std::vector<Eigen::Vector3d> match_points_;
    double                       relo_Pose_[SIZE_POSE];
    Eigen::Matrix3d              drift_correct_r_;
    Eigen::Vector3d              drift_correct_t_;
    Eigen::Vector3d              prev_relo_t_;
    Eigen::Matrix3d              prev_relo_r_;
    Eigen::Vector3d              relo_relative_t_;
    Eigen::Quaterniond           relo_relative_q_;
    double                       relo_relative_yaw_;
};
