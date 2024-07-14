#pragma once
#include <iostream>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

#include "config.h"
#include "utils.h"

using namespace std;

struct track_dim{
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
};


class Trajectory
{
public:
    Trajectory();
    Trajectory(Config& config);
    Trajectory(BBox init_bb, float init_score, int init_timestamp, int label, bool tracking_bb_size, bool bb_as_features, Config& config);
    // ~Trajectory();
    int len();
    void init_parameters();
    void init_trajectory();
    int compute_track_dim();
    void state_prediction(int time_stamp);
    void state_update(BBox bb, float score, int timestamp);
    void filtering();

public:
    bool tracking_features_;
    bool tracking_bb_size_;
    bool bb_as_features_;

    int label_;
    int tracking_dim_;
    int consecutive_missed_num_;

    int init_timestamp_;
    int current_timestamp_;
    int first_updated_timestamp_;
    int last_updated_timestamp_;

    float init_score_;
    float measure_func_covariance_ = 0.001;
    float scanning_interval_ = 0.1;

    Config cfg_;

    vec8x1 init_bb_; // 7

    mat13x13 A;
    mat13x13 Q;
    mat7x7 P;
    mat7x13 B;
    mat13x7 H;
    mat13x13 K;

    Eigen::Matrix3f velo = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f acce = Eigen::Matrix3f::Identity();

    map<int, State> states_;
};