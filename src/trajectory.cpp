#include "trajectory.h"


static float sigmoid(float score)
{
    return 1.0/(1.0 + exp(-float(score)));
}

Trajectory::Trajectory()
{

}

Trajectory::Trajectory(Config& config)
{
    tracking_features_ = true;
    bb_as_features_ = false;
    tracking_bb_size_ = true;

    cfg_ = config;

    consecutive_missed_num_ = 0;
    tracking_dim_ = 0;
    scanning_interval_ = 1./cfg_.LiDAR_scanning_frequency;
}

Trajectory::Trajectory(BBox init_bb,
                       float init_score, int init_timestamp,
                       int label, bool tracking_bb_size, bool bb_as_features,
                       Config& config)
{
    init_bb_ = init_bb;
    init_score_ = init_score;
    init_timestamp_ = init_timestamp;
    label_ = label;

    tracking_features_ = true;
    bb_as_features_ = bb_as_features;
    tracking_bb_size_ = tracking_bb_size;
    tracking_dim_ = 13;

    cfg_ = config;

    consecutive_missed_num_ = 0;
    scanning_interval_ = 1./cfg_.LiDAR_scanning_frequency;

    init_parameters();
    init_trajectory();

    first_updated_timestamp_ = init_timestamp;
    last_updated_timestamp_ = init_timestamp;
}

int Trajectory::len()
{
    return states_.size();
}

int Trajectory::compute_track_dim()
{
    tracking_dim_ = 9;  // x,y,z,vx,vy,vz,ax,ay,az

    if(tracking_bb_size_){
        tracking_dim_ += 4; // w,h,l,yaw
    }

    return tracking_dim_;
}


void Trajectory::init_parameters()
{
    A = Eigen::MatrixXf::Identity(13, 13);
    K = Eigen::MatrixXf::Identity(13, 13);
    Q = Eigen::MatrixXf::Identity(13, 13) * cfg_.state_func_covariance;
    P = Eigen::MatrixXf::Identity(7, 7) * cfg_.measure_func_covariance;

    for (int i = 0; i < 3; ++i) {  
        for (int j = 0; j < B.cols(); ++j) {  
            B(i, j) = A(i, j);
        }  
    }
    
    for (int i = 3; i < B.rows(); ++i) {  
        for (int j = 0; j < B.cols(); ++j) {  
            B(i, j) = A(i+6, j);
        }  
    }

    velo = velo * scanning_interval_;
    acce = acce * 0.5 * scanning_interval_ * scanning_interval_;

    for (int i = 0; i < 3; ++i) {  
        for (int j = 3; j < 6; ++j) {  
            A(i, j) = velo(i, j - 3);
        }  
        for (int j = 6; j < 9; ++j) {  
            A(i, j) = acce(i, j - 6);
        } 
    }
    
    for (int i = 3; i < 6; ++i) {  
        for (int j = 6; j < 9; ++j) {  
            A(i, j) = velo(i - 3, j - 6);
        }  
    }

    H = B.transpose();
    K(3, 0) = scanning_interval_;
    K(4, 1) = scanning_interval_;
    K(5, 2) = scanning_interval_;
}

void Trajectory::init_trajectory() {
    vec7x1 detected_state_template = Eigen::MatrixXf::Zero(7, 1);
    mat13x13 update_covariance_template = Eigen::MatrixXf::Identity(13, 13) * 0.01;

    for(int i = 0; i < 3; i++){
        detected_state_template[i] = init_bb_(i); // #init x,y,z
    }

    if(tracking_bb_size_){
        for(int i = 3; i < 7; i++){
            detected_state_template[i] = init_bb_(i); // #init x,y,z
        }
    }
    else{
        if(tracking_features_){
            for(int i = 3; i < detected_state_template.size(); i++){
                detected_state_template[i] = init_bb_(i); // #init x,y,z
            }
        }
    }

    mat13x13 update_covariance_template_ = update_covariance_template.transpose();
    vec13x1 update_state_template = H * detected_state_template;

    State state;

    state.updated_state = update_state_template;
    state.predicted_state = update_state_template;
    state.detected_state = detected_state_template;
    state.updated_covariance = update_covariance_template;
    state.predicted_covariance = update_covariance_template;
    state.prediction_score = 1;
    state.score = init_score_;
    // object.features = init_features_;

    states_[init_timestamp_] = state;
}

void Trajectory::state_prediction(int time_stamp) {
    int previous_timestamp = time_stamp - 1;

    State previous_state = states_[previous_timestamp];

    vec13x1 previous_predicted_state;
    mat13x13 previous_covariance;

    if (!isEmpty(previous_state.updated_state)) {
        previous_predicted_state = previous_state.updated_state;
        previous_covariance = previous_state.updated_covariance;
    }
    else {
        previous_predicted_state = previous_state.predicted_state;
        previous_covariance = previous_state.predicted_covariance;
    }

    float previous_prediction_score = previous_state.prediction_score;
    float current_prediction_score;

    auto it = states_.find(time_stamp - 1);
    if (it != states_.end()) {
        if (!isEmpty(states_[time_stamp - 1].updated_state)) {
            current_prediction_score = previous_prediction_score * (1 - cfg_.prediction_score_decay * 15);
        }
        else {
            current_prediction_score = previous_prediction_score * (1 - cfg_.prediction_score_decay);
        }
    }
    else {
        current_prediction_score = previous_prediction_score * (1 - cfg_.prediction_score_decay);
    }

    vec13x1 current_predicted_state = A * previous_predicted_state;
    mat13x13 current_predicted_covariance = A * previous_covariance * A.transpose() + Q;

    State new_state;
    new_state.predicted_state = current_predicted_state;
    new_state.predicted_covariance = current_predicted_covariance;
    new_state.prediction_score = current_prediction_score;

    states_[time_stamp] = new_state;
    consecutive_missed_num_ += 1;
}

void Trajectory::state_update(BBox bb, float score, int timestamp) {
    assert(!isEmpty(bb));

    auto it = states_.find(timestamp);
    assert(it != states_.end());

    vec7x1 detected_state_template = Eigen::MatrixXf::Zero(7, 1);

    for (int i = 0; i < 3; i++) {
        detected_state_template(i, 0) = bb[i];
    }

    if (tracking_bb_size_) {
        for (int i = 3; i < 7; i++) {
            detected_state_template(i, 0) = bb[i];
        }
    }

    vec13x1 predicted_state = states_[timestamp].predicted_state;
    mat13x13 predicted_covariance = states_[timestamp].predicted_covariance;

    // KF update
    mat7x7 temp = B * predicted_covariance * B.transpose() + P;
    mat13x7 KF_gain = predicted_covariance * B.transpose() * temp.inverse();
    Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(13, 13);
    vec13x1 updated_state = predicted_state + KF_gain * (detected_state_template - B * predicted_state);
    mat13x13 updated_covariance = (identity - KF_gain * B) * predicted_covariance;

    if (states_.size() == 2) {
        updated_state = H * detected_state_template + K * (H * detected_state_template - states_[timestamp - 1].updated_state);
    }

    states_[timestamp].updated_state = updated_state;
    states_[timestamp].updated_covariance = updated_covariance;
    states_[timestamp].detected_state = detected_state_template;

    if(consecutive_missed_num_ > 1) states_[timestamp].prediction_score = 1;

    else if (!isEmpty(states_[timestamp - 1].updated_state)) {
        states_[timestamp].prediction_score = states_[timestamp].prediction_score + cfg_.prediction_score_decay * 10 * (sigmoid(score));
    }
    else {
        states_[timestamp].prediction_score = states_[timestamp].prediction_score + cfg_.prediction_score_decay * (sigmoid(score));
    }
    
    states_[timestamp].score = score;
    // states_[timestamp].features = features;
    consecutive_missed_num_ = 0;
    current_timestamp_ = timestamp;
    last_updated_timestamp_ = timestamp;
}

void Trajectory::filtering() {
    int window_size = cfg_.LiDAR_scanning_frequency * cfg_.latency;

    float det_num = 0.00001;
    float score_sum = 0.0;

    if (window_size < 0) {
        for (auto& [key, state]: states_) {
            if (state.score != 0) {
                det_num += 1;
                score_sum += state.score;
            }
            if ((first_updated_timestamp_ < current_timestamp_ &&
                 current_timestamp_ < last_updated_timestamp_) &&
                isEmpty(state.updated_state)) {
                state.updated_state = state.predicted_state;
            }
        }

        for (auto & [key, state] : states_) {
            state.score = score_sum / det_num;
        }
    }
    else {
        for (auto & [key, state] : states_) {
            int min_key = key - window_size;
            int max_key = key + window_size;

            for (int i = min_key; i < max_key; i++) {
                auto it = states_.find(i);

                if (it != states_.end()) {
                    if (state.score != 0) {
                        det_num += 1;
                        score_sum += state.score;
                    }
                    if ((first_updated_timestamp_ < current_timestamp_ &&
                        current_timestamp_ < last_updated_timestamp_) &&
                        isEmpty(state.updated_state)) {
                        state.updated_state = state.predicted_state;
                    }
                }
            }

            for (auto& [key, state] : states_) {
                state.score = score_sum / det_num;
            }

            int score = score_sum / det_num;

            if (window_size > 0) {
                states_[key].score = score;
            }
        }
    }
}