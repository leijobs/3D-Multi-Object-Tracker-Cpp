#pragma once

#include <cmath>
#include <string>
#include <fstream>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace fs = std::filesystem;

const float float_pi = 3.14159265358979323846f;

typedef Eigen::Vector4f vec4x1;
typedef Eigen::Matrix<float, 7, 1> vec7x1;
typedef Eigen::Matrix<float, 8, 1> vec8x1;
typedef Eigen::Matrix<float, 13, 1> vec13x1;
typedef Eigen::Matrix<float, 14, 1> vec14x1;
typedef Eigen::Matrix<float, 3, 4> mat3x4;
typedef Eigen::Matrix<float, 7, 7> mat7x7;
typedef Eigen::Matrix<float, 7, 13> mat7x13;
typedef Eigen::Matrix<float, 13, 7> mat13x7;
typedef Eigen::Matrix<float, 13, 13> mat13x13;

typedef Eigen::Matrix<float, 8, 3> Corner3d;
typedef vec8x1 BBox;
typedef std::array<float, 13> Features;

struct Config{
    string dataset_path;
    string detections_path;
    string save_path;
    string tracking_type;

    float measure_func_covariance{};
    float prediction_score_decay{};

    std::vector<int> tracking_seqs;
    int LiDAR_scanning_frequency{};
    int state_func_covariance{};
    int max_prediction_num{};
    int max_prediction_num_for_new_object{};
    float input_score{};
    float init_score{};
    float update_score{};
    float post_score{};
    int latency{};
};

// follow your project
struct Feature {
    int x;
};


struct State {
    float prediction_score;
    float score;

    vec13x1 features;      // None
    vec13x1 updated_state;    // 13 x 1
    vec13x1 predicted_state;  // 13 x 1
    vec7x1 detected_state;   // 7 x 1

    mat13x13 updated_covariance;  // 13 x 13
    mat13x13 predicted_covariance; // 13 x 13

};

struct Label {
    string type;
    int truncated;
    int occluded;
    float alpha;
    float x1, y1, x2, y2;
    float w, h, l;
    float x, y, z;
    float rotation_y;
    float score;

    void clear() {
        type = ""; 
        truncated = 0;
        occluded = 0;
        alpha = 0.0f;
        x1 = y1 = x2 = y2 = 0.0f;
        w = h = l = 0.0f;
        x = y = z = 0.0f;
        rotation_y = 0.0f;
        score = 0.0f;
    }
};