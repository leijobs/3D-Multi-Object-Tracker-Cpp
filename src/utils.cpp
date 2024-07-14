#include "utils.h"

float get_time() {
    auto start = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<float>>(start.time_since_epoch()).count();
}

bool isEmpty(Eigen::MatrixXf mat) {
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if (mat(i, j) != 0) return false;
        }
    }
    return true;
}

Eigen::Vector3f toEigenVector3f(const std::array<float, 3>& arr) {
    return Eigen::Vector3f(arr[0], arr[1], arr[2]);
}

std::array<float, 3> fromEigenVector3f(const Eigen::Vector3f& vec) {
    return {vec[0], vec[1], vec[2]};
}

vector<BBox> convert_bbs_type(vector<BBox>& boxes, const string& input_box_type)
{
    if(input_box_type != "Kitti") return boxes;

    vector<BBox> new_boxes;
    // (h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)
    for(auto & box : boxes){
        BBox bbox;
        bbox << box[3], box[4], box[5] + box[0] / 2,
                box[2], box[1], box[0], 1.5 * float_pi - box[6], box[7];
        new_boxes.push_back(bbox);
    }

    return new_boxes;
}

float get_registration_angle(Eigen::Matrix4f& mat)
{
    float cos_theta = mat(0, 0);
    float sin_theta = mat(1, 0);

    cos_theta = (cos_theta < -1) ? -1 : cos_theta;
    cos_theta = (cos_theta > 1) ? 1 : cos_theta;
    float theta_cos = acos(cos_theta);

    return (sin_theta >= 0) ? theta_cos : 2 * float_pi - theta_cos;
}



void register_bbs(vector<BBox>& boxes, Eigen::Matrix4f& pose)
{
    if (isEmpty(pose)) return;

    float angle = get_registration_angle(pose);

    for (int i = 0; i < boxes.size();i++) {
        Eigen::RowVector4f box_xyz1(boxes[i][0], boxes[i][1], boxes[i][2], 1.0f);
        Eigen::RowVector4f box_world = box_xyz1 * pose.transpose();

        boxes[i][0] = box_world(0);
        boxes[i][1] = box_world(1);
        boxes[i][2] = box_world(2);
        boxes[i][6] += angle;
    }
}


void corners3d_to_img_boxes(mat3x4& P2, Eigen::MatrixXf& corners3d, BBox& new_boxes)
{
    Eigen::MatrixXf corners3d_hom(corners3d.rows(), 4);
    for (int i = 0; i < corners3d.rows(); ++i) {
        corners3d_hom.row(i).head(3) = corners3d.row(i).head(3);
        corners3d_hom(i, 3) = 1.0f;
    }

    Eigen::MatrixXf img_pts = corners3d_hom * P2.transpose();

    for (int i = 0; i < img_pts.rows(); ++i) {
        if (img_pts(i, 2) == 0) continue;

        img_pts(i, 0) /= img_pts(i, 2);
        img_pts(i, 1) /= img_pts(i, 2);
    }

    float min_x = img_pts.row(0).minCoeff();
    float max_x = img_pts.row(0).maxCoeff();
    float min_y = img_pts.row(1).minCoeff();
    float max_y = img_pts.row(1).maxCoeff();

    new_boxes[0] = min(max(min_x, (float)0.0), (float)1242 - 1);
    new_boxes[1] = min(max(min_y, (float)0.0), (float)375 - 1);
    new_boxes[2] = min(max(max_x, (float)0.0), (float)1242 - 1);
    new_boxes[3] = min(max(max_y, (float)0.0), (float)375 - 1);
}

void bb3d_2_bb2d(BBox& boxes, BBox& new_boxes, mat3x4& P2)
{
    float x = boxes[0];
    float y = boxes[1];
    float z = boxes[2];
    float l = boxes[3];
    float w = boxes[4];
    float h = boxes[5];
    float yaw = boxes[6];

    Eigen::MatrixXf pts(8, 4);

    pts <<
        l / 2, 0, w / 2, 1,
        l / 2, 0, -w / 2, 1,
        -l / 2, 0, w / 2, 1,
        -l / 2, 0, -w / 2, 1,
        l / 2, -h, w / 2, 1,
        l / 2, -h, -w / 2, 1,
        -l / 2, -h, w / 2, 1,
        -l / 2, -h, -w / 2, 1;

    Eigen::Matrix4f transform;
    transform <<
              cos(float_pi - yaw), 0, -sin(float_pi - yaw), x,
              0, 1, 0, y,
              sin(float_pi - yaw), 0, cos(float_pi - yaw), z,
              0, 0, 0, 1;

    pts = pts * transform.transpose();
    pts *= P2.transpose();
    corners3d_to_img_boxes(P2, pts, new_boxes);
}

Eigen::Vector3f cam_to_velo(Eigen::Vector4f& data, Eigen::Matrix4f& V2C)
{
    Eigen::Matrix4f V2C_inv = V2C.inverse();
    Eigen::Matrix<float, 3, 4> mat3x4;
    for(int i=0; i< V2C.rows() - 1; i++){
        for(int j=0; j < V2C.cols(); j++){
            mat3x4(i, j) = V2C_inv(i, j);
        }
    }
    return mat3x4 * data;
}

Eigen::Vector3f velo_to_cam(Eigen::Vector4f& data, Eigen::Matrix4f& V2C)
{
    Eigen::Matrix<float, 3, 4> mat3x4;
    for(int i=0; i< V2C.rows() - 1; i++){
        for(int j=0; j < V2C.cols(); j++){
            mat3x4(i, j) = V2C(i, j);
        }
    }
    return mat3x4 * data;
}

bool create_directories(const fs::path& path) {
    try {
        if (!fs::exists(path)) {
            fs::create_directories(path);
            std::cout << "Created directory: " << path << std::endl;
            return true;
        }
        else {
            std::cout << "Directory already exists: " << path << std::endl;
            return true;
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }
}

//Config cfg_from_yaml_file(const string& config_yaml)
//{
//    YAML::Node config = YAML::LoadFile(config_yaml);
//
//    Config cfg;
//
//    std::cout << " ########################## config info ########################## " << std::endl;
//    // dataset_path
//    if (config["dataset_path"]) {
//        cfg.dataset_path = config["dataset_path"].as<std::string>();
//        std::cout << "dataset_path: " << cfg.dataset_path << std::endl;
//    }
//    // detections_path
//    if (config["detections_path"]) {
//        cfg.detections_path = config["detections_path"].as<std::string>();
//        std::cout << "detections_path: " << cfg.detections_path << std::endl;
//    }
//    // save_path
//    if (config["save_path"]) {
//        cfg.save_path = config["save_path"].as<std::string>();
//        std::cout << "save_path: " << cfg.save_path << std::endl;
//    }
//    //tracking_seqs
//    if (config["tracking_seqs"]) {
//        if (config["tracking_seqs"].IsSequence()) {
//            for (const auto& seq : config["tracking_seqs"]) {
//                cfg.tracking_seqs.push_back(seq.as<int>());
//            }
//
//            const int* const_data = cfg.tracking_seqs.data();
//            for (size_t i = 0; i < cfg.tracking_seqs.size(); ++i) {
//                std::cout << const_data[i] << ' ';
//            }
//            std::cout << std::endl;
//        }
//        else {
//            std::cerr << "tracking_seqs is not a sequence." << std::endl;
//        }
//    }
//    else {
//        std::cerr << "tracking_seqs not found in the YAML file." << std::endl;
//    }
//
//    // tracking_type
//    if (config["tracking_type"]) {
//        cfg.tracking_type = config["tracking_type"].as<std::string>();
//        std::cout << "tracking_type: " << cfg.tracking_type << std::endl;
//    }
//
//    // state_func_covariance
//    if (config["state_func_covariance"]) {
//        cfg.state_func_covariance = config["state_func_covariance"].as<int>();
//        std::cout << "state_func_covariance: " << cfg.state_func_covariance << std::endl;
//    }
//
//    //prediction_score_decay
//    if (config["prediction_score_decay"]) {
//        cfg.prediction_score_decay = config["prediction_score_decay"].as<float>();
//        std::cout << "prediction_score_decay: " << cfg.prediction_score_decay << std::endl;
//    }
//
//    // LiDAR_scanning_frequency
//    if (config["LiDAR_scanning_frequency"]) {
//        cfg.LiDAR_scanning_frequency = config["LiDAR_scanning_frequency"].as<int>();
//        std::cout << "LiDAR_scanning_frequency: " << cfg.LiDAR_scanning_frequency << std::endl;
//    }
//
//    //measure_func_covariance
//    if (config["measure_func_covariance"]) {
//        cfg.measure_func_covariance = config["measure_func_covariance"].as<float>();
//        std::cout << "measure_func_covariance: " << cfg.measure_func_covariance << std::endl;
//    }
//
//    // max_prediction_num
//    if (config["max_prediction_num"]) {
//        cfg.max_prediction_num = config["max_prediction_num"].as<int>();
//        std::cout << "max_prediction_num: " << cfg.max_prediction_num << std::endl;
//    }
//
//    // max_prediction_num_for_new_object
//    if (config["max_prediction_num_for_new_object"]) {
//        cfg.max_prediction_num_for_new_object = config["max_prediction_num_for_new_object"].as<int>();
//        std::cout << "max_prediction_num_for_new_object: " << cfg.max_prediction_num_for_new_object << std::endl;
//    }
//
//    // input_score
//    if (config["input_score"]) {
//        cfg.input_score = config["input_score"].as<float>();
//        std::cout << "input_score: " << cfg.input_score << std::endl;
//    }
//
//    // init_score
//    if (config["init_score"]) {
//        cfg.init_score = config["init_score"].as<float>();
//        std::cout << "init_score: " << cfg.init_score << std::endl;
//    }
//
//    // update_score
//    if (config["update_score"]) {
//        cfg.update_score = config["update_score"].as<float>();
//        std::cout << "update_score: " << cfg.update_score << std::endl;
//    }
//
//    // post_score
//    if (config["post_score"]) {
//        cfg.post_score = config["post_score"].as<float>();
//        std::cout << "post_score: " << cfg.post_score << std::endl;
//    }
//
//    // latency
//    if (config["latency"]) {
//        cfg.latency = config["latency"].as<int>();
//        std::cout << "latency: " << cfg.latency << std::endl;
//    }
//    std::cout << " ########################## config info ########################## " << std::endl;
//
//    return cfg;
//}
