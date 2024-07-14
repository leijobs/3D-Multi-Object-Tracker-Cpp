#include "kitti_dataset.h"

#include <utility>

static std::size_t readFloatBinFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "error in loading  " << filename << std::endl;
        return 0;
    }

    file.seekg(0, file.end);
    std::streampos fileSize = file.tellg();

    if (fileSize % sizeof(float) != 0) {
        std::cerr << "file not match float type" << std::endl;
        file.close();
        return 0;
    }

    file.seekg(0, file.beg);
    std::size_t numFloats = fileSize / sizeof(float);
    file.close();
    return numFloats;
}

KittiTrackingDataset::KittiTrackingDataset(const string& dataset_dir, const string& detection_dir, int id)
{
    root_dir = dataset_dir;
    seq_id = id;
    seq_name = get_index(seq_id, 4);
    velo_dir = root_dir + "/velodyne";
    image_dir= root_dir + "/image_02";
    calib_dir = root_dir + "/calib";
    label_dir = root_dir + "/label_02";
    pose_dir = root_dir + "/pose";
    ob_dir = detection_dir;
    length = 0;

    calib_path = calib_dir + "/" + seq_name + ".txt";
    label_path = label_dir + "/" + seq_name + ".txt";
    pose_path = pose_dir + "/" + seq_name + "/" + "pose.txt";
    read_pose();
}

int KittiTrackingDataset::len() {
    if (root_dir.empty()) return 0;
    else {
        // std::cout << "loading calib : " << calib_dir << std::endl;
        try {
            filesystem::directory_iterator list(velo_dir);
            int i = 0;
            for (auto& it : list)
            {
                i++;
            }
            length = i;
            return i;
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "loading calib error: " << e.what() << std::endl;
            return -1;
        }
    }
}

string KittiTrackingDataset::get_index(int index, int totalDigits)
{
    std::stringstream ss;
    ss << std::setw(totalDigits) << std::setfill('0') << index;
    return ss.str();
}

void KittiTrackingDataset::get_item(int idx) {
    seq_sub_name = get_index(idx, 6);
    velo_path = velo_dir + "/" + seq_name + "/" + seq_sub_name + ".bin";
    image_path = image_dir + "/" + seq_name +  "/" + seq_sub_name + ".png";
    ob_path = ob_dir + "/" + seq_name + "/" + seq_sub_name + ".txt";

    read_calib();
    cur_seq.clear();
    cur_seq.index = idx;
    cur_seq.pose = pose_seqs[cur_seq.index];
    read_ob_label();
    if(load_points) read_velodyne();
    if(load_images) read_image();
}

void KittiTrackingDataset::read_calib()
{
    std::string line, key;  
    Eigen::Matrix4f matrix;
  
    std::ifstream file(calib_path);
    if (!file.is_open()) {  
        std::cerr << "Failed to open calib file: " << calib_path << std::endl;
    }  
  
    while (std::getline(file, line)) {  
        std::istringstream iss(line);  
        if (!(iss >> key)) continue; // 读取键（如"R0_rect"）  
  
        if (key == "P2:") {  
            calib.P2.setIdentity(); // 初始化为单位矩阵  
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float value;
                    if (!(iss >> value)) break; // 读取值  
                    calib.P2(i, j) = value;
                }  
            }  
        }
        if (key == "R_rect") {
            calib.R0.setIdentity(); // 初始化为单位矩阵  
            for (int i = 0; i < 3; ++i) {  
                for (int j = 0; j < 3; ++j) {
                    float value;
                    if (!(iss >> value)) break; // 读取值  
                    calib.R0(i, j) = value;
                }  
            }  
        }  
        if (key == "Tr_velo_cam") {
            calib.V2C.setIdentity(); // 初始化为单位矩阵  
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float value;
                    if (!(iss >> value)) break; // 读取值  
                    calib.V2C(i, j) = value;
                }  
            }  
        }  
    }
    calib.V2C = calib.R0 * calib.V2C;
    file.close();  
}

void KittiTrackingDataset::read_velodyne()
{
    std::ifstream file(velo_path, std::ios::binary);  
    if (!file.is_open()) {  
        std::cerr << "Failed to open velo file: " << velo_path << std::endl;
    }  
    
    size_t bin_num = readFloatBinFile(velo_path);
    size_t num_points = bin_num / 4;
    cur_seq.velodyne.resize(int(num_points), 4);
  
    float point[4];  
    for (auto i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(point), sizeof(float) * 4);  
        if (!file) {  
            std::cerr << "Failed to read all points from file." << std::endl;  
            break;  
        }
        cur_seq.velodyne(i, 0) = point[0]; // x
        cur_seq.velodyne(i, 1) = point[1]; // y
        cur_seq.velodyne(i, 2) = point[2]; // z
        cur_seq.velodyne(i, 3) = 1.0; // = 1
    }  
    file.close();  
}

void KittiTrackingDataset::read_image()
{
    cur_seq.image = cv::imread(image_path);
}

void KittiTrackingDataset::read_ob_label()
{
    std::ifstream file(ob_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open label file: " << ob_path << std::endl;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        Label label;

        iss >> label.type >> label.truncated >> label.occluded >> label.alpha
            >> label.x1 >> label.y1 >> label.x2 >> label.y2
            >> label.w >> label.h >> label.l
            >> label.x >> label.y >> label.z
            >> label.rotation_y >> label.score;

        Eigen::Vector4f input_vec(label.x, label.y, label.z, 1.0);
        Eigen::Vector3f output_vec = cam_to_velo(input_vec, calib.V2C);

        vec8x1 obj;
        obj(0, 0) = label.w;
        obj(1, 0) = label.h;
        obj(2, 0) = label.l;
        obj(3, 0) = output_vec(0);
        obj(4, 0) = output_vec(1);
        obj(5, 0) = output_vec(2);
        obj(6, 0) = label.rotation_y;
        obj(7, 0) = label.score;

        cur_seq.objects.emplace_back(obj);
        cur_seq.labels.emplace_back(label);
    }

    file.close();
}

void KittiTrackingDataset::read_tracking_label()
{
    std::ifstream file(label_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open label file: " << label_path << std::endl;
    }
    std::string line;
    while (std::getline(file, line)) {  
        std::istringstream iss(line);  

        Label label;
        
        iss >> label.type >> label.truncated >> label.occluded >> label.alpha 
            >> label.x1 >> label.y1 >> label.x2 >> label.y2  
            >> label.w >> label.h >> label.l 
            >> label.x >> label.y >> label.z 
            >> label.rotation_y >> label.score;  

        vec8x1 obj;
        obj(0, 0) = label.w;
        obj(1, 0) = label.h;
        obj(2, 0) = label.l;
        obj(3, 0) = label.x;
        obj(4, 0) = label.y;
        obj(5, 0) = label.z;
        obj(6, 0) = label.rotation_y;
        obj(7, 0) = label.score;
        
        cur_seq.objects.emplace_back(obj);
        cur_seq.labels.emplace_back(label);
    }  
  
    file.close();  
}

void KittiTrackingDataset::read_pose()
{
    std::ifstream file(pose_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open pose file: " << pose_path << std::endl;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Eigen::MatrixXf pose = Eigen::MatrixXf::Identity(4, 4);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                double value;
                if (!(iss >> value)) break; // 读取值
                pose(i, j) = value;
            }
        }
        pose_seqs.emplace_back(pose);
    }

    file.close();
}

void KittiTrackingDataset::filter_by_score(float score_thresh)
{
    cur_seq.objects.erase(std::remove_if(cur_seq.objects.begin(), cur_seq.objects.end(),
            [score_thresh](const vec8x1& obj) {
            return (obj(7, 0) < score_thresh);
        }), cur_seq.objects.end());
    cur_seq.labels.erase(std::remove_if(cur_seq.labels.begin(), cur_seq.labels.end(),
        [score_thresh](const Label& label) {
            return (label.score < score_thresh);
        }), cur_seq.labels.end());
}

void KittiTrackingDataset::filter_by_type(const string& type)
{
    for(int i=0; i<cur_seq.labels.size(); i++){
        if(type != cur_seq.labels[i].type){
            cur_seq.objects.erase(cur_seq.objects.begin() + i);
            cur_seq.labels.erase(cur_seq.labels.begin() + i);
        }
    }
}
