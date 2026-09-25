/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include <stdexcept>
#include <yaml-cpp/yaml.h>

double INIT_DEPTH;
double MIN_PARALLAX;
Eigen::Vector3d ACC_N, ACC_W;
Eigen::Vector3d GYR_N, GYR_W;
int IMU_NOISE_IS_DENSITY;
double FOCAL_LENGTH = 460.0;
double FREQ;
int DIAGNOSTICS;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

Eigen::Vector3d readNoiseParameter(const cv::FileNode &node, const std::string &name)
{
    Eigen::Vector3d value;
    if (node.empty())
        throw std::runtime_error("missing IMU noise parameter: " + name);

    if (node.isSeq())
    {
        if (node.size() != 3)
            throw std::runtime_error(name + " must be a scalar or a three-axis vector");
        int index = 0;
        for (auto it = node.begin(); it != node.end(); ++it)
            value(index++) = static_cast<double>(*it);
    }
    else
    {
        value.setConstant(static_cast<double>(node));
    }

    if (!value.allFinite() || (value.array() < 0.0).any())
        throw std::runtime_error(name + " must contain finite, non-negative values");
    return value;
}

Eigen::Matrix4d readExtrinsic(const cv::FileNode &node, const std::string &name)
{
    cv::Mat cvTransform;
    node >> cvTransform;
    if (cvTransform.rows != 4 || cvTransform.cols != 4)
        throw std::runtime_error(name + " must be a 4x4 matrix");

    Eigen::Matrix4d transform;
    cv::cv2eigen(cvTransform, transform);
    if (!transform.allFinite())
        throw std::runtime_error(name + " contains non-finite values");

    const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    const double orthogonalityError =
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm();
    if (orthogonalityError > 1e-5 || std::abs(rotation.determinant() - 1.0) > 1e-5 ||
        (transform.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() > 1e-9)
        throw std::runtime_error(name + " is not a valid rigid transform");
    return transform;
}

std::string calibrationPath(const std::string &configFile, const std::string &path)
{
    if (path.empty())
        throw std::runtime_error("calibration path is empty");
    if (path[0] == '/')
        return path;
    const auto slash = configFile.find_last_of('/');
    return (slash == std::string::npos ? "." : configFile.substr(0, slash)) + "/" + path;
}

std::pair<Eigen::Matrix4d, double> readKalibrCamera(const YAML::Node &root, int camera)
{
    const std::string name = "cam" + std::to_string(camera);
    const YAML::Node node = root[name];
    const YAML::Node rows = node["T_cam_imu"];
    if (!rows.IsSequence() || rows.size() != 4)
        throw std::runtime_error(name + " T_cam_imu must have four rows");
    Eigen::Matrix4d T;
    for (int r = 0; r < 4; ++r)
    {
        if (!rows[r].IsSequence() || rows[r].size() != 4)
            throw std::runtime_error(name + " T_cam_imu row must have four values");
        for (int c = 0; c < 4; ++c)
            T(r, c) = rows[r][c].as<double>();
    }
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    if (!T.allFinite() || (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 1e-5 ||
        std::abs(R.determinant() - 1.0) > 1e-5 ||
        (T.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9)
        throw std::runtime_error(name + " T_cam_imu is not a rigid transform");
    const double shift = node["timeshift_cam_imu"].as<double>();
    if (!std::isfinite(shift))
        throw std::runtime_error(name + " timeshift_cam_imu is not finite");
    Eigen::Matrix4d inverse = Eigen::Matrix4d::Identity();
    inverse.block<3, 3>(0, 0) = R.transpose();
    inverse.block<3, 1>(0, 3) = -R.transpose() * T.block<3, 1>(0, 3);
    return {inverse, shift};
}

Eigen::Vector3d readAllanAxes(const cv::FileStorage &source, const char *sensor, const char *field)
{
    Eigen::Vector3d values;
    const char *axes[] = {"x-axis", "y-axis", "z-axis"};
    for (int i = 0; i < 3; ++i)
    {
        const cv::FileNode node = source[sensor][axes[i]][field];
        if (node.empty())
            throw std::runtime_error(std::string("missing Allan ") + sensor + "/" + axes[i] + "/" + field);
        values[i] = static_cast<double>(node);
    }
    if (!values.allFinite() || (values.array() < 0).any())
        throw std::runtime_error(std::string("invalid Allan noise: ") + field);
    return values;
}


template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    RIC.clear();
    TIC.clear();
    CAM_NAMES.clear();

    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        throw std::runtime_error("config file does not exist: " + config_file);
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        throw std::runtime_error("failed to open config file: " + config_file);
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    FREQ = fsSettings["freq"];
    if (!std::isfinite(FREQ) || FREQ < 0.0)
        throw std::runtime_error("freq must be finite and non-negative");
    DIAGNOSTICS = fsSettings["diagnostics"].empty() ? 0 : static_cast<int>(fsSettings["diagnostics"]);
    if (DIAGNOSTICS != 0 && DIAGNOSTICS != 1)
        throw std::runtime_error("diagnostics must be 0 or 1");

    const cv::FileNode camchainNode = fsSettings["kalibr_camchain"];
    std::string camchainFile;
    if (!camchainNode.empty())
    {
        if (!fsSettings["body_T_cam0"].empty() || !fsSettings["body_T_cam1"].empty() ||
            !fsSettings["T_cam_imu0"].empty() || !fsSettings["T_cam_imu1"].empty() ||
            !fsSettings["td"].empty())
            throw std::runtime_error("kalibr_camchain cannot be combined with inline extrinsics or td");
        camchainNode >> camchainFile;
    }

    cv::FileNode focalLengthNode = fsSettings["focal_length"];
    FOCAL_LENGTH = focalLengthNode.empty() ? 460.0 : static_cast<double>(focalLengthNode);
    if (!std::isfinite(FOCAL_LENGTH) || FOCAL_LENGTH <= 0.0)
        throw std::runtime_error("focal_length must be finite and positive");

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        const cv::FileNode allanNode = fsSettings["imu_allan"];
        const cv::FileNode kalibrImuNode = fsSettings["kalibr_imu"];
        if (!allanNode.empty() && !kalibrImuNode.empty())
            throw std::runtime_error("choose imu_allan or kalibr_imu, not both");
        if (!allanNode.empty() || !kalibrImuNode.empty())
        {
            if (!fsSettings["acc_n"].empty() || !fsSettings["acc_w"].empty() ||
                !fsSettings["gyr_n"].empty() || !fsSettings["gyr_w"].empty())
                throw std::runtime_error("imu_allan cannot be combined with acc_n/acc_w/gyr_n/gyr_w");
            std::string sourceFile;
            (allanNode.empty() ? kalibrImuNode : allanNode) >> sourceFile;
            const std::string path = calibrationPath(config_file, sourceFile);
            if (!allanNode.empty())
            {
                cv::FileStorage source(path, cv::FileStorage::READ);
                if (!source.isOpened())
                    throw std::runtime_error("cannot open Allan calibration: " + path);
                ACC_N = readAllanAxes(source, "Acc", "acc_n");
                ACC_W = readAllanAxes(source, "Acc", "acc_w");
                GYR_N = readAllanAxes(source, "Gyr", "gyr_n");
                GYR_W = readAllanAxes(source, "Gyr", "gyr_w");
                ROS_INFO("Allan three-axis densities: %s", path.c_str());
            }
            else
            {
                const YAML::Node imu = YAML::LoadFile(path)["imu0"];
                ACC_N.setConstant(imu["accelerometer_noise_density"].as<double>());
                ACC_W.setConstant(imu["accelerometer_random_walk"].as<double>());
                GYR_N.setConstant(imu["gyroscope_noise_density"].as<double>());
                GYR_W.setConstant(imu["gyroscope_random_walk"].as<double>());
                for (const auto &noise : {ACC_N, ACC_W, GYR_N, GYR_W})
                    if (!noise.allFinite() || (noise.array() < 0).any())
                        throw std::runtime_error("invalid Kalibr IMU density: " + path);
                ROS_INFO("Kalibr IMU scalar densities: %s", path.c_str());
            }
        }
        else
        {
            ACC_N = readNoiseParameter(fsSettings["acc_n"], "acc_n");
            ACC_W = readNoiseParameter(fsSettings["acc_w"], "acc_w");
            GYR_N = readNoiseParameter(fsSettings["gyr_n"], "gyr_n");
            GYR_W = readNoiseParameter(fsSettings["gyr_w"], "gyr_w");
        }
        cv::FileNode noiseDensityNode = fsSettings["imu_noise_is_density"];
        const bool calibratedNoise = !allanNode.empty() || !kalibrImuNode.empty();
        IMU_NOISE_IS_DENSITY = noiseDensityNode.empty() ? (calibratedNoise ? 1 : 0) : static_cast<int>(noiseDensityNode);
        if (IMU_NOISE_IS_DENSITY != 0 && IMU_NOISE_IS_DENSITY != 1)
            throw std::runtime_error("imu_noise_is_density must be 0 or 1");
        if (calibratedNoise && IMU_NOISE_IS_DENSITY != 1)
            throw std::runtime_error("Allan/Kalibr noise must be used as continuous-time densities");
        G.z() = fsSettings["g_norm"];
        if (!std::isfinite(G.z()) || G.z() <= 0.0)
            throw std::runtime_error("g_norm must be finite and positive");
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;

    NUM_OF_CAM = fsSettings["num_of_cam"];
    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
        throw std::runtime_error("num_of_cam must be 1 or 2");
    STEREO = NUM_OF_CAM == 2;
    YAML::Node kalibr;
    const bool hasKalibr = !camchainNode.empty();
    if (hasKalibr)
    {
        const std::string path = calibrationPath(config_file, camchainFile);
        kalibr = YAML::LoadFile(path);
        ROS_INFO("Kalibr T_cam_imu and time offsets: %s", path.c_str());
    }

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (hasKalibr && ESTIMATE_EXTRINSIC == 2)
        throw std::runtime_error("kalibr_camchain requires estimate_extrinsic 0 or 1");
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        const cv::FileNode kalibrMatrix = fsSettings["T_cam_imu0"];
        if (!kalibrMatrix.empty() && !fsSettings["body_T_cam0"].empty())
            throw std::runtime_error("choose T_cam_imu0 or body_T_cam0, not both");
        const Eigen::Matrix4d T = hasKalibr ? readKalibrCamera(kalibr, 0).first :
            (!kalibrMatrix.empty() ? readExtrinsic(kalibrMatrix, "T_cam_imu0").inverse().eval() :
             readExtrinsic(fsSettings["body_T_cam0"], "body_T_cam0"));
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    printf("camera number %d\n", NUM_OF_CAM);


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        const cv::FileNode kalibrMatrix = fsSettings["T_cam_imu1"];
        if (!kalibrMatrix.empty() && !fsSettings["body_T_cam1"].empty())
            throw std::runtime_error("choose T_cam_imu1 or body_T_cam1, not both");
        const Eigen::Matrix4d T = hasKalibr ? readKalibrCamera(kalibr, 1).first :
            (!kalibrMatrix.empty() ? readExtrinsic(kalibrMatrix, "T_cam_imu1").inverse().eval() :
             readExtrinsic(fsSettings["body_T_cam1"], "body_T_cam1"));
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
        if (hasKalibr)
            ROS_INFO("Kalibr cam1 time shift %.9f s; VINS uses cam0 only", readKalibrCamera(kalibr, 1).second);
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    if (!fsSettings["T_cam_imu0"].empty() && fsSettings["td"].empty())
        throw std::runtime_error("T_cam_imu0 requires an explicit cam0 td");
    TD = hasKalibr ? readKalibrCamera(kalibr, 0).second : static_cast<double>(fsSettings["td"]);
    if (!std::isfinite(TD))
        throw std::runtime_error("td must be finite");
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    if (ROW <= 0 || COL <= 0)
        throw std::runtime_error("image_width and image_height must be positive");
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    if (!fout)
        ROS_WARN("cannot write trajectory: %s", VINS_RESULT_PATH.c_str());

    fsSettings.release();
}
