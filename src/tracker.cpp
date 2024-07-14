#include "tracker.h"


Tracker3D::Tracker3D(string box_type, bool tracking_features, bool tracking_bb_size, bool bb_as_features, Config& cfg)
{
    box_type_ = box_type;
    tracking_features_ = tracking_features;
    tracking_bb_size_ = tracking_bb_size;
    bb_as_features_ = bb_as_features;
    cfg_ = cfg;
}

void Tracker3D::get_measurement(const data_per_seq& measurement) {
    measurement_ = measurement;
}


void Tracker3D::tracking()
{
    // save as local param
    current_bbs_ = measurement_.objects;
    // current_features_ = features;
    current_pose_ = measurement_.pose;
    current_timestamp_ = measurement_.index;

    trajectores_prediction();

    // checking
    if(!current_bbs_.empty()){
        current_bbs_ = convert_bbs_type(current_bbs_, box_type_);
        register_bbs(current_bbs_, current_pose_);
        vector<int> ids = association();
        trajectories_update_init(ids);
    }
}

void Tracker3D::trajectores_prediction()
{
    if (active_trajectories.empty()){
        return;
    }
    else{
        vector<int> dead_track_id;

        for (auto& [key, traj] : active_trajectories) {
            if(traj.consecutive_missed_num_ >= cfg_.max_prediction_num){
                dead_track_id.push_back(key);
                continue;
            }

            if (traj.states_.size() - traj.consecutive_missed_num_ == 1 &&
                traj.states_.size()>= cfg_.max_prediction_num_for_new_object){
                dead_track_id.push_back(key);
            }
            traj.state_prediction(current_timestamp_);
        }

        for(auto& id : dead_track_id){
            Trajectory traj = active_trajectories[id];
            active_trajectories.erase(id);
            dead_trajectories[id] = traj;
        }
    }
}

void Tracker3D::compute_cost_map()
{
    vector<vec14x1> all_predictions;
    vector<vec13x1> all_detections;
    all_ids.clear();

    for(const auto& [key, traj] : active_trajectories){
        all_ids.push_back(key);

        map<int, State> states = traj.states_;
        float predicted_score = states[current_timestamp_].prediction_score;
        vec13x1 predicted_state = states[current_timestamp_].predicted_state;

        vec14x1 single_prediction;
        single_prediction << predicted_state(0), predicted_state(1), predicted_state(2), predicted_state(3),
                             predicted_state(4), predicted_state(5), predicted_state(6), predicted_state(7),
                             predicted_state(8), predicted_state(9), predicted_state(10), predicted_state(11),
                             predicted_state(12), predicted_score;
        all_predictions.push_back(single_prediction);
    }

    for(int i = 0; i < current_bbs_.size(); i++){
        Trajectory new_tra = Trajectory(current_bbs_[i],
                                        current_bbs_[i](7, 0),
                                        current_timestamp_,
                                        1,
                                        tracking_features_,
                                        bb_as_features_,
                                        cfg_);

        vec13x1 state = new_tra.states_[current_timestamp_].predicted_state;
        all_detections.push_back(state);
    }

    int det_size = all_detections.size();
    int pred_size = all_predictions.size();

    if (cost_mat.rows() != det_size || cost_mat.cols() != pred_size) {
        cost_mat.resize(det_size, pred_size);
    }

    for(int i = 0; i < det_size;i++){
        for(int j = 0; j < pred_size; j++){
            float sum_data = 0;
            for(int k = 0; k < 3;k++){
                sum_data += ((all_detections[i](k) - all_predictions[j](k))
                            * (all_detections[i](k) - all_predictions[j](k)));
            }
            cost_mat(i, j) = sqrt(sum_data) * all_predictions[j](13);
        }
    }
}

vector<int> Tracker3D::association()
{
    vector<int> ids;
    if (active_trajectories.empty()){

        for(int i=0; i < current_bbs_.size(); i++){
            ids.push_back(label_seed_);
            label_seed_ += 1;
        }
    }
    else{
        compute_cost_map();
        for(int i = 0; i < current_bbs_.size(); i++){
            float min_val = 100001;
            int min_idx = 0;

            for (int j = 0; j < cost_mat.cols(); ++j) {
                if (cost_mat(i, j) < min_val) {
                    min_val = cost_mat(i, j);
                    min_idx = j;
                }
            }

            if(min_val < 2.0){
                ids.push_back(all_ids[min_idx]);
                for(int j = 0; j < cost_mat.rows(); j++){
                    cost_mat(j, min_idx) = 100000;
                }

            }
            else{
                ids.push_back(label_seed_);
                label_seed_ += 1;
            }
        }
    }
    return ids;
}

void Tracker3D::trajectories_update_init(vector<int>& ids)
{
    assert(ids.size() == current_bbs_.size());
    valid_bbs.clear();
    valid_ids.clear();

    for(int i = 0; i < current_bbs_.size(); i++){
        int label = ids[i];
        BBox box = current_bbs_[i];
        float score = current_bbs_[i](7);

        auto it = active_trajectories.find(label);

        if(it != active_trajectories.end() && score > cfg_.update_score){
            active_trajectories[label].state_update(box, score, current_timestamp_);
            valid_bbs.push_back(box);
            valid_ids.push_back(label);
        }
        else if(score > cfg_.init_score) {
            // new trajectory
            Trajectory new_tra = Trajectory(box, score, current_timestamp_,
                                            label, tracking_bb_size_, bb_as_features_, cfg_);
            active_trajectories[label] = new_tra;
            valid_bbs.push_back(box);
            valid_ids.push_back(label);
        }
        else{
            continue;
        }
    }
}

void Tracker3D::post_processing()
{
    for(auto& [key, traj] : dead_trajectories)
    {
        traj.filtering();
        all_trajectories[key] = traj;
    }
    for(auto& [key, traj] : active_trajectories)
    {
        traj.filtering();
        all_trajectories[key] = traj;
    }
}
