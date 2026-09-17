/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    processThreadRunning = false;
    resetGeneration = 0;
    clearState();
}

Estimator::~Estimator()
{
    shutdown();
}

void Estimator::shutdown()
{
    processThreadRunning = false;
    if (processThread.joinable())
        processThread.join();
}

void Estimator::clearState()
{
    mProcess.lock();
    resetGeneration.fetch_add(1);
    {
        std::lock_guard<std::mutex> propagation_lock(mPropagate);
        std::lock_guard<std::mutex> tracker_lock(mTracker);
        {
            std::lock_guard<std::mutex> buffer_lock(mBuf);
            while(!accBuf.empty())
                accBuf.pop();
            while(!gyrBuf.empty())
                gyrBuf.pop();
            while(!featureBuf.empty())
                featureBuf.pop();
        }

        prevTime = -1;
        curTime = 0;
        openExEstimation = 0;
        initP = Eigen::Vector3d(0, 0, 0);
        initR = Eigen::Matrix3d::Identity();
        inputImageCnt = 0;
        initFirstPoseFlag = false;

        for (int i = 0; i < WINDOW_SIZE + 1; i++)
        {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
            dt_buf[i].clear();
            linear_acceleration_buf[i].clear();
            angular_velocity_buf[i].clear();

            if (pre_integrations[i] != nullptr)
            {
                delete pre_integrations[i];
                pre_integrations[i] = nullptr;
            }
            pre_integrations[i] = nullptr;
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d::Zero();
            ric[i] = Matrix3d::Identity();
        }

        first_imu = false,
        sum_of_back = 0;
        sum_of_front = 0;
        frame_count = 0;
        solver_flag = INITIAL;
        initial_timestamp = 0;
        for (auto &entry : all_image_frame)
            delete entry.second.pre_integration;
        all_image_frame.clear();

        if (tmp_pre_integration != nullptr)
        {
            delete tmp_pre_integration;
            tmp_pre_integration = nullptr;
        }
        if (last_marginalization_info != nullptr)
        {
            delete last_marginalization_info;
            last_marginalization_info = nullptr;
        }

        tmp_pre_integration = nullptr;
        last_marginalization_info = nullptr;
        last_marginalization_parameter_blocks.clear();

        f_manager.clearState();

        failure_occur = 0;
    }
    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    {
        std::lock_guard<std::mutex> propagation_lock(mPropagate);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = TIC[i];
            ric[i] = RIC[i];
            cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
        }
        f_manager.setRic(ric);
        ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
        ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
        ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
        td = TD;
        g = G;
        cout << "set g " << g.transpose() << endl;
    }
    {
        std::lock_guard<std::mutex> tracker_lock(mTracker);
        featureTracker.readIntrinsicParameter(CAM_NAMES);
    }

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThreadRunning = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                delete tmp_pre_integration;
                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    const uint64_t generation = resetGeneration.load();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    bool enqueue_frame;
    {
        std::lock_guard<std::mutex> tracker_lock(mTracker);
        inputImageCnt++;
        enqueue_frame = !MULTIPLE_THREAD || inputImageCnt % 2 == 0;
        if(_img1.empty())
            featureFrame = featureTracker.trackImage(t, _img);
        else
            featureFrame = featureTracker.trackImage(t, _img, _img1);
        //printf("featureTracker time: %f\n", featureTrackerTime.toc());

        if (SHOW_TRACK)
        {
            cv::Mat imgTrack = featureTracker.getTrackImage();
            pubTrackImage(imgTrack, t);
        }
    }

    bool enqueued = false;
    if (enqueue_frame)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        if (resetGeneration.load() == generation)
        {
            featureBuf.push(make_pair(generation, make_pair(t, featureFrame)));
            enqueued = true;
        }
    }
    if (!MULTIPLE_THREAD && enqueued)
    {
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    const uint64_t generation = resetGeneration.load();
    {
        std::lock_guard<std::mutex> lock(mBuf);
        if (resetGeneration.load() != generation)
            return;
        accBuf.push(make_pair(t, linearAcceleration));
        gyrBuf.push(make_pair(t, angularVelocity));
        //printf("input imu with time %f \n", t);
    }

    std::lock_guard<std::mutex> propagation_lock(mPropagate);
    if (solver_flag == NON_LINEAR)
    {
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    ROS_ERROR("deprecated at VINS-Fusion");
    assert(0);
    mBuf.lock();
    const uint64_t generation = resetGeneration.load();
    featureBuf.push(make_pair(generation, make_pair(t, featureFrame)));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty() || gyrBuf.empty() || accBuf.size() != gyrBuf.size())
    {
        printf("IMU buffers are empty or inconsistent\n");
        while (!accBuf.empty())
            accBuf.pop();
        while (!gyrBuf.empty())
            gyrBuf.pop();
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first && t1 <= gyrBuf.back().first)
    {
        while (!accBuf.empty() && accBuf.front().first <= t0)
        {
            // std::cout << "t_imu: " << std::fixed << accBuf.front().first << "  t_0: " << std::fixed << t0 << "   gyr_buf size: " << gyrBuf.size() << std::endl;
            // std::cout << "1) acc pop" << std::endl;
            accBuf.pop();
            // std::cout << "1) gyr pop" << std::endl;
            gyrBuf.pop();
        }
        if (accBuf.empty() || gyrBuf.empty())
            return false;
        while (!accBuf.empty() && accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            // std::cout << "2) acc pop" << std::endl;
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            // std::cout << "2) gyr pop" << std::endl;
            gyrBuf.pop();
        }
        if (accBuf.empty() || gyrBuf.empty())
            return false;
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    std::lock_guard<std::mutex> lock(mBuf);
    if(!accBuf.empty() && !gyrBuf.empty() &&
       t <= accBuf.back().first && t <= gyrBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (!MULTIPLE_THREAD || processThreadRunning)
    {
        // cout << "[processMeasurements]  loop - start" << endl;

        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        uint64_t feature_generation = 0;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        bool has_feature = false;
        {
            std::lock_guard<std::mutex> lock(mBuf);
            if (!featureBuf.empty())
            {
                feature_generation = featureBuf.front().first;
                feature = featureBuf.front().second;
                has_feature = true;
            }
        }
        if(has_feature)
        {
            // cout << "1" << endl;
            curTime = feature.first + td;
            // std::cout << "t0: " << std::fixed << curTime << std::endl;
            while (!MULTIPLE_THREAD || processThreadRunning)
            {
                if (resetGeneration.load() != feature_generation)
                    break;
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            if (MULTIPLE_THREAD && !processThreadRunning)
                break;
            // cout << "2" << endl;
            bool interval_ready = true;
            mProcess.lock();
            {
                std::lock_guard<std::mutex> lock(mBuf);
                if (resetGeneration.load() != feature_generation ||
                    featureBuf.empty() ||
                    featureBuf.front().first != feature_generation ||
                    featureBuf.front().second.first != feature.first)
                    interval_ready = false;
                else
                {
                    if(USE_IMU)
                    {
                        // cout << "2-1)" << endl;
                        interval_ready = getIMUInterval(prevTime, curTime, accVector, gyrVector);
                        // cout << "2-2)" << endl;
                    }
                    if (interval_ready)
                        featureBuf.pop();
                }
            }
            if (!interval_ready)
            {
                mProcess.unlock();
                if (!MULTIPLE_THREAD)
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            // cout << "3" << endl;
            if(USE_IMU)
            {
                bool imu_interval_valid = true;
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    // getIMUInterval keeps the first sample after curTime for
                    // interpolation. ZOH must not hold that future sample over
                    // the beginning of the next image interval.
                    const bool advance_held_measurement =
                        !IMU_PREINTEGRATION_ENABLE ||
                        accVector[i].first <= curTime;
                    if (!processIMU(accVector[i].first, dt,
                                    accVector[i].second, gyrVector[i].second,
                                    advance_held_measurement))
                    {
                        imu_interval_valid = false;
                        break;
                    }
                }
                if (!imu_interval_valid)
                {
                    mProcess.unlock();
                    ROS_ERROR("Rejected IMU interval; restarting estimator at current image time");
                    clearState();
                    setParameter();
                    continue;
                }
            }
            // cout << "4" << endl;

            const bool image_valid = processImage(feature.second, feature.first);
            prevTime = curTime;

            if (!image_valid)
            {
                mProcess.unlock();
                ROS_ERROR("Rejected image update; restarting estimator at current image time");
                clearState();
                setParameter();
                continue;
            }

            // cout << "5" << endl;

            printStatistics(*this, 0);

            std_msgs::msg::Header header;
            header.frame_id = "world";

            int sec_ts = (int)feature.first;
            uint nsec_ts = (uint)((feature.first - sec_ts) * 1e9);
            header.stamp.sec = sec_ts;
            header.stamp.nanosec = nsec_ts;

            pubOdometry(*this, header);
            // cout << "5-1" << endl;
            pubKeyPoses(*this, header);
            // cout << "5-2" << endl;
            pubCameraPose(*this, header);
            // cout << "5-3" << endl;
            pubPointCloud(*this, header);
            // cout << "5-4" << endl;
            pubKeyframe(*this);
            // cout << "5-5" << endl;
            pubTF(*this, header);
            // cout << "5-6" << endl;
            mProcess.unlock();


            // cout << "6" << endl;

            // assert(0);
        }
        // cout << "[processMeasurements]  loop - end" << endl;

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


bool Estimator::processIMU(double t, double dt,
                           const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity,
                           bool advance_held_measurement)
{
    if (!std::isfinite(t) || !std::isfinite(dt) || dt <= 0.0 ||
        !linear_acceleration.allFinite() || !angular_velocity.allFinite())
        return false;

    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        const int j = frame_count;
        Matrix3d next_R = Rs[j];
        Vector3d next_P = Ps[j];
        Vector3d next_V = Vs[j];
        bool prediction_valid = true;
        if (IMU_PREINTEGRATION_ENABLE)
        {
            prediction_valid = equivariant::propagateWorldStateZoh(
                dt, acc_0, gyr_0, Bas[j], Bgs[j], g,
                next_R, next_V, next_P);
        }
        else
        {
            const Vector3d un_acc_0 = next_R * (acc_0 - Bas[j]) - g;
            const Vector3d un_gyr =
                0.5 * (gyr_0 + angular_velocity) - Bgs[j];
            next_R *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
            const Vector3d un_acc_1 =
                next_R * (linear_acceleration - Bas[j]) - g;
            const Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            next_P += dt * next_V + 0.5 * dt * dt * un_acc;
            next_V += dt * un_acc;
            prediction_valid = next_R.allFinite() && next_P.allFinite() &&
                               next_V.allFinite();
        }
        if (!prediction_valid)
        {
            ROS_ERROR("State prediction rejected IMU interval at %.9f", t);
            return false;
        }

        IntegrationBase::Checkpoint window_checkpoint =
            pre_integrations[frame_count]->checkpoint();
        if (!pre_integrations[frame_count]->push_back(
                dt, linear_acceleration, angular_velocity) ||
            !tmp_pre_integration->push_back(
                dt, linear_acceleration, angular_velocity))
        {
            pre_integrations[frame_count]->restore(
                std::move(window_checkpoint));
            ROS_ERROR("Atomic IMU propagation rejected interval at %.9f", t);
            return false;
        }

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        Rs[j] = next_R;
        Ps[j] = next_P;
        Vs[j] = next_V;
    }
    if (advance_held_measurement)
    {
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    return true;
}

bool Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{


    cout << std::fixed << header << endl;

    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    if (!all_image_frame.insert(make_pair(header, imageframe)).second)
    {
        ROS_ERROR("Duplicate image timestamp %.9f", header);
        return false;
    }
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }


    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    if (optimization())
                    {
                        updateLatestStates();
                        solver_flag = NON_LINEAR;
                        if (!slideWindow())
                            return false;
                        ROS_INFO("Initialization finish!");
                    }
                    else
                    {
                        ROS_WARN("Initialization solve rejected");
                        if (!slideWindow())
                            return false;
                    }
                }
                else
                    if (!slideWindow())
                        return false;
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                if (!solveGyroscopeBias(all_image_frame, Bgs))
                {
                    ROS_ERROR("Gyroscope-bias initialization failed");
                    if (!slideWindow())
                        return false;
                    return true;
                }
                std::vector<std::pair<IntegrationBase *, std::unique_ptr<IntegrationBase>>> updates;
                for (int j = 1; j <= frame_count; ++j)
                {
                    std::unique_ptr<IntegrationBase> candidate =
                        pre_integrations[j]->clone();
                    if (!candidate ||
                        !candidate->repropagate(Bas[j - 1], Bgs[j - 1]))
                        break;
                    updates.emplace_back(
                        pre_integrations[j], std::move(candidate));
                }
                if (updates.size() != static_cast<size_t>(frame_count))
                {
                    ROS_ERROR("Gyroscope-bias initialization repropagation failed");
                    if (!slideWindow())
                        return false;
                    return true;
                }
                for (auto &update : updates)
                    if (!update.first->commitFrom(*update.second))
                        return false;
                if (optimization())
                {
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    if (!slideWindow())
                        return false;
                    ROS_INFO("Initialization finish!");
                }
                else
                {
                    ROS_WARN("Initialization solve rejected");
                    if (!slideWindow())
                        return false;
                }
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            const bool initial_solve_usable = optimization();

            if(frame_count == WINDOW_SIZE)
            {
                if (initial_solve_usable && optimization())
                {
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    if (!slideWindow())
                        return false;
                    ROS_INFO("Initialization finish!");
                }
                else
                {
                    ROS_WARN("Stereo initialization solve rejected");
                    if (!slideWindow())
                        return false;
                }
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

        // optimization
        TicToc t_solve;
        if (!optimization())
        {
            ROS_ERROR("Solver result rejected; restarting estimator instead of advancing without a prior");
            return false;
        }
        ROS_INFO("solver costs: %f [ms]", t_solve.toc());

        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }
            

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            ROS_WARN("system reboot!");
            return false;
        }

        if (!slideWindow())
            return false;
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
    return true;
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g = Vector3d::Zero();
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    std::vector<std::pair<IntegrationBase *, std::unique_ptr<IntegrationBase>>> updates;
    for (int j = 1; j <= frame_count; ++j)
    {
        std::unique_ptr<IntegrationBase> candidate =
            pre_integrations[j]->clone();
        if (!candidate ||
            !candidate->repropagate(Bas[j - 1], Bgs[j - 1]))
            return false;
        updates.emplace_back(pre_integrations[j], std::move(candidate));
    }
    for (auto &update : updates)
        if (!update.first->commitFrom(*update.second))
            return false;

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // cout << Ps[i].x() << " " << Ps[i].y() << " " << Ps[i].z() << endl;
        // cout << "--------" << endl;

        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

bool Estimator::repropagateImuPreintegrations(bool &repropagated)
{
    // v1.2 adds one atomic bias repropagation and one bounded re-solve before
    // marginalization; v1.1 kept the original linearization for the window.
    repropagated = false;
    if (!USE_IMU)
        return true;

    std::vector<std::pair<IntegrationBase *, std::unique_ptr<IntegrationBase>>> updates;
    for (int j = 1; j <= frame_count; ++j)
    {
        IntegrationBase *integration = pre_integrations[j];
        if (!integration)
            return false;
        const bool bias_changed =
            (Bas[j - 1] - integration->linearized_ba).norm() >
                BIAS_ACC_THRESHOLD ||
            (Bgs[j - 1] - integration->linearized_bg).norm() >
                BIAS_GYR_THRESHOLD;
        if (bias_changed)
        {
            std::unique_ptr<IntegrationBase> candidate = integration->clone();
            if (!candidate ||
                !candidate->repropagate(Bas[j - 1], Bgs[j - 1]))
                return false;
            updates.emplace_back(integration, std::move(candidate));
        }
    }

    if (tmp_pre_integration)
    {
        const bool bias_changed =
            (Bas[frame_count] - tmp_pre_integration->linearized_ba).norm() >
                BIAS_ACC_THRESHOLD ||
            (Bgs[frame_count] - tmp_pre_integration->linearized_bg).norm() >
                BIAS_GYR_THRESHOLD;
        if (bias_changed)
        {
            std::unique_ptr<IntegrationBase> candidate =
                tmp_pre_integration->clone();
            if (!candidate ||
                !candidate->repropagate(Bas[frame_count], Bgs[frame_count]))
                return false;
            updates.emplace_back(tmp_pre_integration, std::move(candidate));
        }
    }

    for (auto &update : updates)
        if (!update.first->commitFrom(*update.second))
            return false;
    repropagated = !updates.empty();
    return true;
}

bool Estimator::optimization()
{
    return optimizationWithBiasRetry(1);
}

bool Estimator::optimizationWithBiasRetry(int retries_remaining)
{
    TicToc t_whole, t_prepare;
    if (USE_IMU)
    {
        for (int j = 1; j <= frame_count; ++j)
        {
            if (pre_integrations[j]->sum_dt <= 10.0 &&
                !pre_integrations[j]->isValidForFactor())
            {
                ROS_ERROR("Refusing optimization with invalid IMU preintegration %d", j);
                return false;
            }
        }
    }
    vector2double();

    std::unique_ptr<ceres::Problem> problem(new ceres::Problem());
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem->AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem->AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem->SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem->AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem->SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem->AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem->SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem->AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem->AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem->AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem->AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem->AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    if (USE_GPU_CERES)
        options.dense_linear_algebra_library_type = ceres::CUDA;

    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;


    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost))
    {
        ROS_ERROR("Ceres solution rejected: %s", summary.BriefReport().c_str());
        return false;
    }

    double2vector();
    if (retries_remaining > 0)
    {
        bool repropagated = false;
        if (!repropagateImuPreintegrations(repropagated))
        {
            ROS_ERROR("Bias repropagation failed after optimization");
            return false;
        }
        if (repropagated)
        {
            problem.reset();
            return optimizationWithBiasRetry(retries_remaining - 1);
        }
    }
    else if (USE_IMU && solver_flag == NON_LINEAR)
    {
        // Keep marginalization consistent with the final biases without
        // allowing an unbounded solve/repropagation loop.
        bool final_repropagated = false;
        if (!repropagateImuPreintegrations(final_repropagated))
        {
            ROS_ERROR("Final bias repropagation failed after bounded re-solve");
            return false;
        }
        if (final_repropagated)
            ROS_DEBUG("Committed final bias repropagation without another solve");
    }
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return true;
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        if (!marginalization_info->preMarginalize())
        {
            ROS_ERROR("Marginalization factor evaluation failed; aborting prior update");
            delete marginalization_info;
            return false;
        }
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        problem.reset();
        if (!marginalization_info->marginalize())
        {
            ROS_ERROR("Marginalization failed; aborting prior update");
            delete marginalization_info;
            return false;
        }
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
        {
            delete last_marginalization_info;
            last_marginalization_info = nullptr;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            if (!marginalization_info->preMarginalize())
            {
                ROS_ERROR("Marginalization factor evaluation failed; aborting prior update");
                delete marginalization_info;
                return false;
            }
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            problem.reset();
            if (!marginalization_info->marginalize())
            {
                ROS_ERROR("Marginalization failed; aborting prior update");
                delete marginalization_info;
                return false;
            }
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
            {
                delete last_marginalization_info;
                last_marginalization_info = nullptr;
            }
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
    return true;
}

bool Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = nullptr;
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            std::unique_ptr<IntegrationBase> merged_preintegration;
            if (USE_IMU)
            {
                merged_preintegration =
                    pre_integrations[frame_count - 1]->clone();
                if (!merged_preintegration)
                {
                    ROS_ERROR("Failed to prepare atomic IMU interval merge");
                    return false;
                }
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); ++i)
                {
                    if (!merged_preintegration->push_back(
                            dt_buf[frame_count][i],
                            linear_acceleration_buf[frame_count][i],
                            angular_velocity_buf[frame_count][i]))
                    {
                        ROS_ERROR("Failed to merge IMU intervals while sliding window");
                        return false;
                    }
                }
                if (!pre_integrations[frame_count - 1]->commitFrom(
                        *merged_preintegration))
                {
                    ROS_ERROR("Failed to commit atomic IMU interval merge");
                    return false;
                }
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                dt_buf[frame_count - 1].insert(
                    dt_buf[frame_count - 1].end(),
                    dt_buf[frame_count].begin(), dt_buf[frame_count].end());
                linear_acceleration_buf[frame_count - 1].insert(
                    linear_acceleration_buf[frame_count - 1].end(),
                    linear_acceleration_buf[frame_count].begin(),
                    linear_acceleration_buf[frame_count].end());
                angular_velocity_buf[frame_count - 1].insert(
                    angular_velocity_buf[frame_count - 1].end(),
                    angular_velocity_buf[frame_count].begin(),
                    angular_velocity_buf[frame_count].end());

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = nullptr;
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
    return true;
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    if (!std::isfinite(dt) || dt <= 0.0 || !linear_acceleration.allFinite() ||
        !angular_velocity.allFinite())
        return;

    Eigen::Matrix3d next_R = latest_Q.toRotationMatrix();
    Eigen::Vector3d next_P = latest_P;
    Eigen::Vector3d next_V = latest_V;
    bool prediction_valid = true;
    if (IMU_PREINTEGRATION_ENABLE)
    {
        prediction_valid = equivariant::propagateWorldStateZoh(
            dt, latest_acc_0, latest_gyr_0, latest_Ba, latest_Bg, g,
            next_R, next_V, next_P);
    }
    else
    {
        const Eigen::Vector3d un_acc_0 =
            next_R * (latest_acc_0 - latest_Ba) - g;
        const Eigen::Vector3d un_gyr =
            0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
        next_R *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        const Eigen::Vector3d un_acc_1 =
            next_R * (linear_acceleration - latest_Ba) - g;
        const Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        next_P += dt * next_V + 0.5 * dt * dt * un_acc;
        next_V += dt * un_acc;
        prediction_valid = next_R.allFinite() && next_P.allFinite() &&
                           next_V.allFinite();
    }
    if (!prediction_valid)
        return;

    latest_time = t;
    latest_Q = Eigen::Quaterniond(next_R).normalized();
    latest_P = next_P;
    latest_V = next_V;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
