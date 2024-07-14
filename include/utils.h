#pragma once
#include <string>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>

#include"config.h"

using namespace std;
using namespace Eigen;


bool isEmpty(Eigen::MatrixXf mat);
float get_time();

Eigen::Vector3f toEigenVector3f(const std::array<float, 3>& arr);

std::array<float, 3> fromEigenVector3f(const Eigen::Vector3f& vec);

vector<BBox> convert_bbs_type(vector<BBox>& boxes, const string& input_box_type);

void register_bbs(vector<BBox>& boxes, Eigen::Matrix4f& pose);

float get_registration_angle(Eigen::Matrix4f& mat);

void corners3d_to_img_boxes(mat3x4& P2, Eigen::MatrixXf& corners3d, BBox& new_boxes);

void bb3d_2_bb2d(BBox& boxes, BBox& new_boxes, mat3x4& P2);

bool create_directories(const fs::path& path);

Config cfg_from_yaml_file(const string& config_yaml);

Eigen::Vector3f cam_to_velo(Eigen::Vector4f& data, Eigen::Matrix4f& V2C);
Eigen::Vector3f velo_to_cam(Eigen::Vector4f& data, Eigen::Matrix4f& V2C);