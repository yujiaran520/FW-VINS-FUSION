/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "globalOpt.h"
#include "Factors.h"

#include <array>
#include <cmath>

namespace
{
builtin_interfaces::msg::Time timeFromSeconds(double time)
{
    const int64_t nanoseconds = static_cast<int64_t>(std::llround(time * 1e9));
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
    return stamp;
}
} // namespace

GlobalOptimization::GlobalOptimization()
{
    initGPS = false;
    newGPS = false;
    stopOptimization = false;
    WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    lastP = Eigen::Vector3d::Zero();
    lastQ = Eigen::Quaterniond::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{
    {
        std::lock_guard<std::mutex> lock(mPoseMap);
        stopOptimization = true;
    }
    optimizationCondition.notify_one();
    if (threadOpt.joinable())
        threadOpt.join();
}

void GlobalOptimization::GPS2XYZ(double latitude, double longitude, double altitude, double* xyz)
{
    if (!initGPS)
    {
        geoConverter.Reset(latitude, longitude, altitude);
        initGPS = true;
    }
    geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
}

void GlobalOptimization::inputOdom(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ)
{
    std::lock_guard<std::mutex> lock(mPoseMap);
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(),
                             OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;

    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP +
                              WGPS_T_WVIO.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = timeFromSeconds(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = lastP.x();
    pose_stamped.pose.position.y = lastP.y();
    pose_stamped.pose.position.z = lastP.z();
    pose_stamped.pose.orientation.x = lastQ.x();
    pose_stamped.pose.orientation.y = lastQ.y();
    pose_stamped.pose.orientation.z = lastQ.z();
    pose_stamped.pose.orientation.w = lastQ.w();
    global_path.header = pose_stamped.header;
    global_path.poses.push_back(pose_stamped);
}

void GlobalOptimization::getGlobalResult(Eigen::Vector3d &odomP,
                                         Eigen::Quaterniond &odomQ,
                                         nav_msgs::msg::Path &path)
{
    std::lock_guard<std::mutex> lock(mPoseMap);
    odomP = lastP;
    odomQ = lastQ;
    path = global_path;
}

void GlobalOptimization::inputGPS(double t, double latitude, double longitude,
                                  double altitude,
                                  const Eigen::Vector3d &positionStdDev)
{
    {
        std::lock_guard<std::mutex> lock(mPoseMap);
        double xyz[3];
        GPS2XYZ(latitude, longitude, altitude, xyz);
        vector<double> gpsPosition{xyz[0], xyz[1], xyz[2],
                                   positionStdDev.x(), positionStdDev.y(),
                                   positionStdDev.z()};
        GPSPositionMap[t] = gpsPosition;
        newGPS = true;
    }
    optimizationCondition.notify_one();
}

void GlobalOptimization::optimize()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(mPoseMap);
        optimizationCondition.wait(lock, [this]
        {
            return newGPS || stopOptimization;
        });
        if (stopOptimization)
            return;

        newGPS = false;
        printf("global optimization\n");

        const size_t length = localPoseMap.size();
        if (length == 0)
            continue;

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5;
        ceres::Solver::Summary summary;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        ceres::LocalParameterization *local_parameterization =
            new ceres::QuaternionParameterization();

        std::vector<std::array<double, 3>> t_array(length);
        std::vector<std::array<double, 4>> q_array(length);
        auto iter = globalPoseMap.begin();
        for (size_t i = 0; i < length; ++i, ++iter)
        {
            t_array[i] = {iter->second[0], iter->second[1], iter->second[2]};
            q_array[i] = {iter->second[3], iter->second[4], iter->second[5],
                          iter->second[6]};
            problem.AddParameterBlock(q_array[i].data(), 4, local_parameterization);
            problem.AddParameterBlock(t_array[i].data(), 3);
        }

        size_t i = 0;
        for (auto iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); ++iterVIO, ++i)
        {
            auto iterVIONext = std::next(iterVIO);
            if (iterVIONext != localPoseMap.end())
            {
                Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                wTi.block<3, 3>(0, 0) =
                    Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4],
                                       iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                wTi.block<3, 1>(0, 3) =
                    Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                wTj.block<3, 3>(0, 0) =
                    Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4],
                                       iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                wTj.block<3, 1>(0, 3) =
                    Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1],
                                    iterVIONext->second[2]);
                const Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                const Eigen::Quaterniond iQj(iTj.block<3, 3>(0, 0));
                const Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);

                ceres::CostFunction *vio_function = RelativeRTError::Create(
                    iPj.x(), iPj.y(), iPj.z(), iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                    0.1, 0.01);
                problem.AddResidualBlock(vio_function, nullptr, q_array[i].data(),
                                         t_array[i].data(), q_array[i + 1].data(),
                                         t_array[i + 1].data());
            }

            const auto iterGPS = GPSPositionMap.find(iterVIO->first);
            if (iterGPS != GPSPositionMap.end())
            {
                ceres::CostFunction *gps_function = TError::Create(
                    iterGPS->second[0], iterGPS->second[1], iterGPS->second[2],
                    iterGPS->second[3], iterGPS->second[4], iterGPS->second[5]);
                problem.AddResidualBlock(gps_function, loss_function, t_array[i].data());
            }
        }

        ceres::Solve(options, &problem, &summary);

        iter = globalPoseMap.begin();
        for (size_t index = 0; index < length; ++index, ++iter)
        {
            vector<double> globalPose{t_array[index][0], t_array[index][1], t_array[index][2],
                                      q_array[index][0], q_array[index][1], q_array[index][2],
                                      q_array[index][3]};
            iter->second = globalPose;
            if (index == length - 1)
            {
                Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                const double t = iter->first;
                WVIO_T_body.block<3, 3>(0, 0) =
                    Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
                                       localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                WVIO_T_body.block<3, 1>(0, 3) =
                    Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                WGPS_T_body.block<3, 3>(0, 0) =
                    Eigen::Quaterniond(globalPose[3], globalPose[4], globalPose[5],
                                       globalPose[6]).toRotationMatrix();
                WGPS_T_body.block<3, 1>(0, 3) =
                    Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                WGPS_T_WVIO = WGPS_T_body * WVIO_T_body.inverse();
            }
        }
        updateGlobalPath();
    }
}

void GlobalOptimization::updateGlobalPath()
{
    global_path.poses.clear();
    for (const auto &globalPose : globalPoseMap)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = timeFromSeconds(globalPose.first);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = globalPose.second[0];
        pose_stamped.pose.position.y = globalPose.second[1];
        pose_stamped.pose.position.z = globalPose.second[2];
        pose_stamped.pose.orientation.w = globalPose.second[3];
        pose_stamped.pose.orientation.x = globalPose.second[4];
        pose_stamped.pose.orientation.y = globalPose.second[5];
        pose_stamped.pose.orientation.z = globalPose.second[6];
        global_path.poses.push_back(pose_stamped);
    }
    if (!global_path.poses.empty())
        global_path.header = global_path.poses.back().header;
}
