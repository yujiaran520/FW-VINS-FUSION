/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include <ceres/internal/config.h>
#include <opencv2/core/cuda.hpp>
#include <cmath>
#include <stdexcept>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
int IMU_PREINTEGRATION_ENABLE = 0;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD = 0.1;
double BIAS_GYR_THRESHOLD = 0.01;
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
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    if (USE_GPU || USE_GPU_ACC_FLOW)
    {
#ifdef GPU_MODE
        const int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_device_count <= 0)
            throw std::runtime_error("GPU tracking requested, but OpenCV found no CUDA device");

        cv::cuda::setDevice(0);
        const cv::cuda::DeviceInfo device_info(0);
        if (!device_info.isCompatible())
            throw std::runtime_error("GPU tracking requested, but CUDA device 0 is incompatible with this OpenCV build");
        ROS_INFO("OpenCV CUDA enabled on device 0: %s", device_info.name());
#else
        throw std::runtime_error("GPU tracking requested, but VINS was built without VINS_ENABLE_GPU");
#endif
    }

    if (USE_GPU_CERES)
    {
#ifdef CERES_NO_CUDA
        throw std::runtime_error("Ceres CUDA requested, but Ceres was built without CUDA");
#else
        ROS_INFO("Ceres CUDA dense linear algebra enabled");
#endif
    }

    USE_IMU = fsSettings["imu"];
    const cv::FileNode equivariant_node = fsSettings["equivariant_preintegration_enable"];
    IMU_PREINTEGRATION_ENABLE = equivariant_node.empty() ? 0 : static_cast<int>(equivariant_node);
    IMU_PREINTEGRATION_ENABLE = IMU_PREINTEGRATION_ENABLE != 0;
    ROS_INFO("Equivariant IMU preintegration: %s",
             IMU_PREINTEGRATION_ENABLE ? "enabled" : "disabled");
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        std::string noise_semantics;
        fsSettings["imu_noise_semantics"] >> noise_semantics;
        if (noise_semantics != "continuous_time_density")
            throw std::runtime_error(
                "imu_noise_semantics must be continuous_time_density");
        const auto valid_density = [](double value) {
            return std::isfinite(value) && value > 0.0;
        };
        if (!valid_density(ACC_N) || !valid_density(GYR_N) ||
            !valid_density(ACC_W) || !valid_density(GYR_W))
            throw std::runtime_error(
                "IMU noise densities must be finite and strictly positive");
        ROS_INFO("Continuous IMU noise densities acc=%g gyr=%g acc_bias=%g gyr_bias=%g",
                 ACC_N, GYR_N, ACC_W, GYR_W);
        G.z() = fsSettings["g_norm"];
        const cv::FileNode bias_acc_node =
            fsSettings["bias_acc_repropagation_threshold"];
        const cv::FileNode bias_gyr_node =
            fsSettings["bias_gyr_repropagation_threshold"];
        if (!bias_acc_node.empty())
            BIAS_ACC_THRESHOLD = static_cast<double>(bias_acc_node);
        if (!bias_gyr_node.empty())
            BIAS_GYR_THRESHOLD = static_cast<double>(bias_gyr_node);
        if (!std::isfinite(BIAS_ACC_THRESHOLD) || BIAS_ACC_THRESHOLD <= 0.0 ||
            !std::isfinite(BIAS_GYR_THRESHOLD) || BIAS_GYR_THRESHOLD <= 0.0)
            throw std::runtime_error(
                "bias repropagation thresholds must be finite and strictly positive");
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
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

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


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
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
