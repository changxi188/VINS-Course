#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[]) : Rs_(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric_[i].setIdentity();
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric_[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature_.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto& it : feature_)
    {
        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(
    int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& image, double td)
{
    LOG(INFO) << "usefull feature number before add new feature : " << getFeatureCount()
              << ", all feature number : " << feature_.size();
    double parallax_sum = 0;
    int    parallax_num = 0;
    last_track_num_     = 0;
    for (auto& id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int  feature_id = id_pts.first;
        auto it         = find_if(feature_.begin(), feature_.end(),
                                  [feature_id](const FeaturePerId& it) { return it.feature_id == feature_id; });

        if (it == feature_.end())
        {
            feature_.push_back(FeaturePerId(feature_id, frame_count));
            feature_.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num_++;
        }
    }

    LOG(INFO) << "usefull feature number after add new feature : " << getFeatureCount()
              << ", all feature number : " << feature_.size();

    if (frame_count < 2 || last_track_num_ < 20)
    {
        return true;
    }

    for (auto& it_per_id : feature_)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        LOG(INFO) << "parallax_sum: " << parallax_sum << ", parallax_num: " << parallax_num;
        LOG(INFO) << "current parallax: " << parallax_sum / parallax_num;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    // ROS_DEBUG("debug show");
    for (auto& it : feature_)
    {
        assert(it.feature_per_frame.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        // ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto& j : it.feature_per_frame)
        {
            // ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ", j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorresponding(int frame_count_l,
                                                                                          int frame_count_r)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto& it : feature_)
    {
        if (it.start_frame > frame_count_l || it.endFrame() < frame_count_r)
        {
            continue;
        }
        Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
        int             idx_l = frame_count_l - it.start_frame;
        int             idx_r = frame_count_r - it.start_frame;

        a = it.feature_per_frame[idx_l].point;

        b = it.feature_per_frame[idx_r].point;

        corres.push_back(std::make_pair(a, b));
    }
    return corres;
}

void FeatureManager::setDepth(const Eigen::VectorXd& x)
{
    int feature_index = -1;
    for (auto& it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // ROS_INFO("feature_ id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame,
        // it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature_.erase(it);
    }
}

void FeatureManager::clearDepth(const Eigen::VectorXd& x)
{
    int feature_index = -1;
    for (auto& it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

Eigen::VectorXd FeatureManager::getDepthVector()
{
    Eigen::VectorXd dep_vec(getFeatureCount());
    int             feature_index = -1;
    for (auto& it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic[], Eigen::Matrix3d ric_[])
{
    for (auto& it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int             svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d             t0 = Ps[imu_i] + Rs_[imu_i] * tic[0];
        Eigen::Matrix3d             R0 = Rs_[imu_i] * ric_[0];
        P0.leftCols<3>()               = Eigen::Matrix3d::Identity();
        P0.rightCols<1>()              = Eigen::Vector3d::Zero();

        for (auto& it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d             t1 = Ps[imu_j] + Rs_[imu_j] * tic[0];
            Eigen::Matrix3d             R1 = Rs_[imu_j] * ric_[0];
            Eigen::Vector3d             t  = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d             R  = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>()      = R.transpose();
            P.rightCols<1>()     = -R.transpose() * t;
            Eigen::Vector3d f    = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double          svd_method = svd_V[2] / svd_V[3];
        // it_per_id->estimated_depth = -b / A;
        // it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        // it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier()
{
    // ROS_BREAK();
    return;
    int i = -1;
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature_.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P)
{
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature_.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i   = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j   = new_R.transpose() * (w_pts_i - new_P);
                double          dep_j   = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature_ after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature_.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature_.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature_.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId& it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame& frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame& frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double          ans = 0;
    Eigen::Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric_[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric_[camera_id_i] * p_i;
    p_i_comp     = p_i;
    double dep_i = p_i(2);
    double u_i   = p_i(0) / dep_i;
    double v_i   = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp   = p_i_comp(0) / dep_i_comp;
    double v_i_comp   = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = std::max(ans, sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
