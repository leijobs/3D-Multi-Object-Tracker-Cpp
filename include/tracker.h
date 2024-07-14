#pragma once
#include <iostream>
#include <Eigen/Dense>

#include "trajectory.h"
#include "kitti_dataset.h"

using namespace std;
using namespace Eigen;


class Tracker3D{
public:
    Tracker3D(string box_type, bool tracking_features, bool tracking_bb_size, bool bb_as_features, Config& cfg);
    void get_measurement(const data_per_seq& measurement);
    void tracking();
    void trajectores_prediction();
    void compute_cost_map();
    vector<int> association();
    void trajectories_update_init(vector<int>& ids);
    void post_processing();

public:
    bool tracking_features_; // if tracking the features
    bool tracking_bb_size_;
    bool bb_as_features_; // if tracking the bbs

    int state_func_covariance_{};
    int label_seed_ = 0;
    int current_timestamp_;

    float scanning_interval_;

    string box_type_; // box type, available box type "OpenPCDet", "Kitti", "Waymo"

    Config cfg_;
    data_per_seq measurement_;

    vector<BBox> valid_bbs;
    vector<int> valid_ids;

    map<int, Trajectory> all_trajectories;
    map<int, Trajectory> active_trajectories;
    map<int, Trajectory> dead_trajectories;

    vector<BBox> current_bbs_;        // array(N,7) or array(N，7*k), 3D bounding boxes or 3D tracklets
    vector<BBox> updated_bbs_;
    vector<Features> current_features_;  // array(N,k), the features of boxes or tracklets
    vector<float> current_scores_;       // the detection score of boxes or tracklets
    Eigen::Matrix4f current_pose_;       // array(4,4), pose matrix to global scene

    vector<int> all_ids;
    Eigen::MatrixXf cost_mat;
};

