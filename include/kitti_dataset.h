#pragma once
#include <map>
#include <string>
#include <iomanip>
#include<filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <algorithm>

#include "config.h"
#include "utils.h"

using namespace std;

struct Calib {
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 4, 4> R0;
    Eigen::Matrix<float, 4, 4> V2C;
};

struct data_per_seq {
public:
    int index;                       // index
    cv::Mat image;                   // image
    vector<Label> labels;            // label
    vector<vec8x1> objects;          // object
    Eigen::Matrix<float, 4, 4> pose; // pose
    Eigen::MatrixXf velodyne;        // velodyne

    data_per_seq() {
        clear();
    }
    ~data_per_seq() {
        clear();
    }
    void clear() {
        index = 0;
        pose.setConstant(0);
        labels.clear();
        objects.clear();
        velodyne.setConstant(0);
    }
};

class KittiTrackingDataset
{
public:
    explicit KittiTrackingDataset(const string& dataset_dir, const string& detection_dir, int id);
    int len();
    void get_item(int idx);
    string get_index(int index, int totalDigits);
    void read_calib();
    void read_velodyne();
    void read_image();
    void read_pose();
    void read_ob_label();
    void read_tracking_label();

    void filter_by_score(float score_thresh);
    void filter_by_type(const string& type);

public:
    bool load_images;
    bool load_points;

    int seq_id;
    int length;

    string seq_name;
    string seq_sub_name;

    string root_dir;
    string velo_dir;
    string image_dir;
    string calib_dir;
    string label_dir;
    string pose_dir;
    string ob_dir;

    string pose_path;
    string velo_path;
    string image_path;
    string calib_path;
    string label_path;
    string ob_path;

    Calib calib;
    data_per_seq cur_seq;
    vector<Eigen::Matrix<float, 4, 4>> pose_seqs;
};
