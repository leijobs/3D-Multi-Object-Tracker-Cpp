#define _USE_MATH_DEFINES
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "kitti_dataset.h"
#include "tracker.h"
#include "config.h"
#include "utils.h"

using namespace std;


map<int, Trajectory> track_one_seq(Config& cfg, int seq_id, float all_time, float all_num)
{
    float start_time = get_time();
    Tracker3D tracker = Tracker3D("Kitti", false, true, false, cfg);
    KittiTrackingDataset dataset = KittiTrackingDataset(cfg.dataset_path, cfg.detections_path, seq_id);

    // run seq
    for (int i = 0; i < dataset.len(); i++){
        // load dataset seq
        dataset.get_item(i);
        // filter labels
        dataset.filter_by_score(cfg.input_score);
        dataset.filter_by_type(cfg.tracking_type);

        // load measurement to tracker
        tracker.get_measurement(dataset.cur_seq);
        // tracking
        tracker.tracking();

        float end_time = get_time();
        all_time += end_time - start_time;
        all_num += 1;
    }
    tracker.post_processing();

    return tracker.all_trajectories;
}

void save_one_seq(Config& cfg, const map<int, Trajectory>& all_trajectories, int seq_id, float all_time, float frame_num){
    string tracking_type = "Car";
    KittiTrackingDataset dataset = KittiTrackingDataset(cfg.dataset_path, cfg.detections_path, seq_id);

    map<int, map<int, vec14x1>> frame_first_dict;
    for (const auto& [key1 ,traj] : all_trajectories) {
        for (const auto& [key2, state] : traj.states_) {
            if (isEmpty(state.updated_state)) continue;
            vec14x1 expanded_state;
            expanded_state << state.updated_state(0), state.updated_state(1), state.updated_state(2), state.updated_state(3), state.updated_state(4),
                              state.updated_state(5), state.updated_state(6), state.updated_state(7), state.updated_state(8), state.updated_state(9),
                              state.updated_state(10), state.updated_state(11), state.updated_state(12), state.score;

            auto it = frame_first_dict.find(key2);
            if (it != frame_first_dict.end()) {
                frame_first_dict[key2][key1] = expanded_state;
            }
            else {
                frame_first_dict[key2][0] = expanded_state;
            }
        }
    }

    // output to txt
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << seq_id;
    string seq_txt = cfg.save_path + "/" + ss.str() + ".txt";
    std::ofstream outfile(seq_txt);
    if (!outfile.is_open()) {
        std::cerr << "fail to open txt" << std::endl;
    }

    for (int i = 0; i < dataset.len(); i++){
        dataset.get_item(i);

        mat3x4 P2 = dataset.calib.P2;
        Eigen::Matrix4f V2C = dataset.calib.V2C;
        Eigen::Matrix4f new_pose = dataset.cur_seq.pose.inverse();


        auto it = frame_first_dict.find(i);
        if (it != frame_first_dict.end()) {
            for (const auto& [ob_id, obejct] : frame_first_dict[i]) {
                vec4x1 vec4f; 
                vec8x1 box, box2d;

                vector<vec8x1> box_template;
                vec8x1 mat = Matrix<float, 8, 1>::Ones();
                box_template.push_back(mat);

                for (int idx = 0; idx < 3; idx++){ box_template[0](idx) = obejct(idx); }
                for (int idx = 3; idx < 7; idx++) { box_template[0](idx) = obejct(idx+6); }

                register_bbs(box_template, new_pose);

                box(6) = -box(6) - M_PI / 2;
                box(2) -= box(5) / 2;
                for (int i = 0; i < 4; ++i) {
                    vec4f[i] = box[i];
                }
                Eigen::Vector3f res = velo_to_cam(vec4f, V2C);
                box(0) = res.x();
                box(1) = res.y();
                box(2) = res.z();

                bb3d_2_bb2d(box, box2d, P2);

                std::stringstream ss;
                ss << i << " " << ob_id << " " << tracking_type << " " << 
                    box2d(0) << " " << box2d(1) << " " << box2d(2) << " " << box2d(3) << " " << 
                    box(5) << " " << box(4) << " " << box(3) << " " << 
                    box(0) << " " << box(1) << " " << box(2) << " " << box(6) << " " << obejct(13);

                outfile << ss.str() << std::endl;
            }
        }
    }
    outfile.close();
    cout << "finished, save to :" << seq_txt << endl;
}

void tracking_val_seq(const string& yaml_file)
{
    Config cfg = cfg_from_yaml_file(yaml_file);

    cout << "config file:" << yaml_file << endl;
    create_directories(cfg.save_path);

    float all_time = 0.0;
    float all_num = 0;

    vector<int> seq_list = cfg.tracking_seqs;

    for(auto& seq_id : cfg.tracking_seqs){
        map<int, Trajectory> all_trajectories = track_one_seq(cfg, seq_id, all_time, all_num);
        save_one_seq(cfg, all_trajectories, seq_id, all_time, all_num);
    }

     cout << "Tracking time: " << all_time << endl;
     cout << "Tracking frames: " << all_num << endl;
     cout << "Tracking FPS:" << all_num /all_time << endl;
     cout << "Tracking ms:" << all_time/ all_num << endl;
}


int main()
{
    string yaml_file = "/home/hosico/JustDoit/3d-mot-cpp/config/global/second_iou_mot.yaml";
    tracking_val_seq(yaml_file);
}
