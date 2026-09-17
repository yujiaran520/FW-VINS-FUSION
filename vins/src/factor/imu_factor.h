/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
#include <rcpputils/asserts.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../estimator/parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
#include <algorithm>
#include <cstdint>

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR


// v1.1 replaced v1.0's center-difference Jacobians with the analytic blocks
// below; v1.2 additionally caches fixed preintegration terms and rejects stale
// cache revisions after bias repropagation.
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration)
        : pre_integration(_pre_integration), cache_valid_(false),
          preintegration_revision_(0), equivariant_dt_(0.0)
    {
        if (!pre_integration || !pre_integration->isValidForFactor())
            return;
        preintegration_revision_ = pre_integration->revision();
        cache_valid_ = computeSqrtInformation(
            pre_integration->covariance, sqrt_information_);
        if (cache_valid_ && pre_integration->usesEquivariantPreintegration())
        {
            const auto &equivariant_preintegration =
                *pre_integration->equivariant_preintegration;
            equivariant_dt_ = pre_integration->sum_dt;
            equivariant_upsilon_ = equivariant_preintegration.upsilon();
            equivariant_bias_hat_ = equivariant_preintegration.biasHat();
            equivariant_bias_correction_ =
                equivariant_preintegration.biasCorrectionJacobian();
        }
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        if (!cache_valid_ || !residuals ||
            !pre_integration->isValidForFactor() ||
            pre_integration->revision() != preintegration_revision_)
            return false;
        const int block_sizes[4] = {7, 9, 7, 9};
        for (int block = 0; block < 4; ++block)
        {
            if (!parameters[block])
                return false;
            for (int index = 0; index < block_sizes[block]; ++index)
                if (!std::isfinite(parameters[block][index]))
                    return false;
        }

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

//Eigen::Matrix<double, 15, 15> Fd;
//Eigen::Matrix<double, 15, 12> Gd;

//Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
//Eigen::Quaterniond pQj = Qi * delta_q;
//Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
//Eigen::Vector3d pBaj = Bai;
//Eigen::Vector3d pBgj = Bgi;

//Vi + Qi * delta_v - g * sum_dt = Vj;
//Qi * delta_q = Qj;

//delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
//delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
//delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        if (jacobians && pre_integration->usesEquivariantPreintegration())
        {
            return evaluateEquivariantJacobians(
                Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj,
                sqrt_information_, residual, jacobians);
        }

        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                             Pj, Qj, Vj, Baj, Bgj);
        if (!residual.allFinite())
            return false;
        residual = sqrt_information_ * residual;
        if (!residual.allFinite())
            return false;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_information_ * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
                //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_information_ * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_j = sqrt_information_ * jacobian_pose_j;

                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_information_ * jacobian_speedbias_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }
        }

        if (jacobians)
        {
            for (int block = 0; block < 4; ++block)
            {
                if (!jacobians[block])
                    continue;
                const Eigen::Map<const Eigen::VectorXd> values(
                    jacobians[block], 15 * block_sizes[block]);
                if (!values.allFinite())
                    return false;
            }
        }
        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    IntegrationBase* pre_integration;

  private:
    static bool computeSqrtInformation(
        const Eigen::Matrix<double, 15, 15> &input_covariance,
        Eigen::Matrix<double, 15, 15> &sqrt_info)
    {
        using Mat15 = Eigen::Matrix<double, 15, 15>;
        Mat15 covariance = 0.5 * (input_covariance + input_covariance.transpose());
        if (!covariance.allFinite())
            return false;

        Eigen::LLT<Mat15> llt(covariance);
        if (llt.info() != Eigen::Success)
        {
            const double scale = std::max(1e-12,
                covariance.diagonal().cwiseAbs().maxCoeff());
            double jitter = scale * 1e-12;
            for (int attempt = 0; attempt < 6 && llt.info() != Eigen::Success; ++attempt)
            {
                Mat15 regularized = covariance;
                regularized.diagonal().array() += jitter;
                llt.compute(regularized);
                jitter *= 10.0;
            }
        }
        if (llt.info() != Eigen::Success)
            return false;

        sqrt_info = llt.matrixL().solve(Mat15::Identity());
        return sqrt_info.allFinite();
    }

    bool evaluateEquivariantJacobians(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
        const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
        const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
        const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
        const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj,
        const Eigen::Matrix<double, 15, 15> &sqrt_info,
        Eigen::Ref<Eigen::Matrix<double, 15, 1>> weighted_residual,
        double **jacobians) const
    {
        using Vec10 = equivariant::Preintegration::Vec10;
        using Mat10 = equivariant::Preintegration::Mat10;
        const double dt = equivariant_dt_;
        const Eigen::Matrix3d Ri_transpose = Qi.inverse().toRotationMatrix();
        const Eigen::Vector3d predicted_p =
            Pi + Vi * dt - 0.5 * G * dt * dt;
        const Eigen::Vector3d predicted_v = Vi - G * dt;
        const Vec10 bias_i = IntegrationBase::makeEquivariantBias(Bai, Bgi);
        const Vec10 bias_j = IntegrationBase::makeEquivariantBias(Baj, Bgj);
        const Mat10 &bias_correction = equivariant_bias_correction_;
        const Vec10 correction =
            bias_correction * (bias_i - equivariant_bias_hat_);
        const equivariant::Gal3 corrected =
            equivariant::Gal3::exp(correction) * equivariant_upsilon_;
        const equivariant::Gal3 truth(
            Ri_transpose * Qj.toRotationMatrix(),
            Ri_transpose * (Vj - predicted_v),
            Ri_transpose * (Pj - predicted_p), dt);
        const equivariant::Gal3 error = truth * corrected.inverse();
        const Vec10 nav_error = equivariant::Gal3::log(error);
        const Mat10 inverse_left_jacobian =
            equivariant::Gal3::inverseLeftJacobian(nav_error);
        const Mat10 truth_adjoint = truth.adjoint();
        const Mat10 error_adjoint = error.adjoint();
        const Vec10 bias_delta = bias_j - bias_i;
        const Vec10 transported_bias_delta = truth_adjoint * bias_delta;
        const Vec10 bias_error =
            -inverse_left_jacobian * transported_bias_delta;
        Eigen::Matrix<double, 15, 1> unweighted_residual;
        unweighted_residual.segment<3>(O_P) = nav_error.segment<3>(6);
        unweighted_residual.segment<3>(O_R) = nav_error.segment<3>(0);
        unweighted_residual.segment<3>(O_V) = nav_error.segment<3>(3);
        unweighted_residual.segment<3>(O_BA) = bias_error.segment<3>(3);
        unweighted_residual.segment<3>(O_BG) = bias_error.segment<3>(0);
        weighted_residual = sqrt_info * unweighted_residual;
        if (!weighted_residual.allFinite())
            return false;

        // H maps a perturbation of Log(error) to the derivative of the
        // inverse-left-Jacobian term in the geometrically coupled bias error.
        Mat10 bias_error_log_jacobian;
        for (int axis = 0; axis < 10; ++axis)
        {
            Vec10 direction = Vec10::Zero();
            direction(axis) = 1.0;
            const Mat10 left_jacobian_derivative =
                equivariant::Gal3::leftJacobianDirectionalDerivative(
                    nav_error, direction);
            bias_error_log_jacobian.col(axis) =
                -inverse_left_jacobian * left_jacobian_derivative * bias_error;
        }
        const Mat10 corrected_bias_left_jacobian =
            equivariant::Gal3::leftJacobian(correction) * bias_correction;

        const auto residualDerivative =
            [&](const Vec10 &truth_left_perturbation,
                const Vec10 &corrected_left_perturbation,
                const Vec10 &bias_delta_perturbation)
                -> Eigen::Matrix<double, 15, 1>
            {
                const Vec10 error_left_perturbation =
                    truth_left_perturbation -
                    error_adjoint * corrected_left_perturbation;
                const Vec10 nav_derivative =
                    inverse_left_jacobian * error_left_perturbation;
                const Vec10 transported_bias_derivative =
                    equivariant::Gal3::algebraAdjoint(
                        truth_left_perturbation) * transported_bias_delta +
                    truth_adjoint * bias_delta_perturbation;
                const Vec10 bias_derivative =
                    bias_error_log_jacobian * nav_derivative -
                    inverse_left_jacobian * transported_bias_derivative;
                Eigen::Matrix<double, 15, 1> derivative;
                derivative.segment<3>(O_P) = nav_derivative.segment<3>(6);
                derivative.segment<3>(O_R) = nav_derivative.segment<3>(0);
                derivative.segment<3>(O_V) = nav_derivative.segment<3>(3);
                derivative.segment<3>(O_BA) = bias_derivative.segment<3>(3);
                derivative.segment<3>(O_BG) = bias_derivative.segment<3>(0);
                const Eigen::Matrix<double, 15, 1> weighted_derivative =
                    sqrt_info * derivative;
                return weighted_derivative;
            };

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            for (int axis = 0; axis < 3; ++axis)
            {
                Vec10 truth_perturbation = Vec10::Zero();
                truth_perturbation.segment<3>(6) =
                    -Ri_transpose.col(axis);
                J.col(axis) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());

                truth_perturbation.setZero();
                truth_perturbation(axis) = -1.0;
                J.col(axis + 3) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());
            }
            if (!J.allFinite())
                return false;
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[1]);
            J.setZero();
            for (int axis = 0; axis < 3; ++axis)
            {
                Vec10 truth_perturbation = Vec10::Zero();
                truth_perturbation.segment<3>(3) =
                    -Ri_transpose.col(axis);
                J.col(axis) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());

                Vec10 bias_perturbation = Vec10::Zero();
                bias_perturbation(axis + 3) = 1.0;
                J.col(axis + 3) = residualDerivative(
                    Vec10::Zero(),
                    corrected_bias_left_jacobian * bias_perturbation,
                    -bias_perturbation);

                bias_perturbation.setZero();
                bias_perturbation(axis) = 1.0;
                J.col(axis + 6) = residualDerivative(
                    Vec10::Zero(),
                    corrected_bias_left_jacobian * bias_perturbation,
                    -bias_perturbation);
            }
            if (!J.allFinite())
                return false;
        }

        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[2]);
            J.setZero();
            for (int axis = 0; axis < 3; ++axis)
            {
                Vec10 truth_perturbation = Vec10::Zero();
                truth_perturbation.segment<3>(6) = Ri_transpose.col(axis);
                J.col(axis) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());

                Vec10 right_rotation = Vec10::Zero();
                right_rotation(axis) = 1.0;
                truth_perturbation = truth_adjoint * right_rotation;
                J.col(axis + 3) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());
            }
            if (!J.allFinite())
                return false;
        }

        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[3]);
            J.setZero();
            for (int axis = 0; axis < 3; ++axis)
            {
                Vec10 truth_perturbation = Vec10::Zero();
                const Eigen::Vector3d velocity_perturbation =
                    Ri_transpose.col(axis);
                truth_perturbation.segment<3>(3) = velocity_perturbation;
                truth_perturbation.segment<3>(6) =
                    -dt * velocity_perturbation;
                J.col(axis) = residualDerivative(
                    truth_perturbation, Vec10::Zero(), Vec10::Zero());

                Vec10 bias_perturbation = Vec10::Zero();
                bias_perturbation(axis + 3) = 1.0;
                J.col(axis + 3) = residualDerivative(
                    Vec10::Zero(), Vec10::Zero(), bias_perturbation);

                bias_perturbation.setZero();
                bias_perturbation(axis) = 1.0;
                J.col(axis + 6) = residualDerivative(
                    Vec10::Zero(), Vec10::Zero(), bias_perturbation);
            }
            if (!J.allFinite())
                return false;
        }
        return true;
    }

    bool cache_valid_;
    std::uint64_t preintegration_revision_;
    Eigen::Matrix<double, 15, 15> sqrt_information_;
    double equivariant_dt_;
    equivariant::Gal3 equivariant_upsilon_;
    equivariant::Preintegration::Vec10 equivariant_bias_hat_;
    equivariant::Preintegration::Mat10 equivariant_bias_correction_;

};
