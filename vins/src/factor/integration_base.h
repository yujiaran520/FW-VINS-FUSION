/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../utility/utility.h"
#include "../estimator/parameters.h"
#include "../IMU_core/equivariant_preintegration.h"

#include <ceres/ceres.h>
#include <cmath>
#include <cstdint>
#include <memory>
using namespace Eigen;

// v1.2 extends the v1.1 preintegration path with continuous-density noise
// discretization plus checkpoint/clone/commit support for atomic propagation.
class IntegrationBase
{
  public:
    struct Checkpoint
    {
        double dt;
        Eigen::Vector3d acc_0, gyr_0, acc_1, gyr_1;
        Eigen::Vector3d linearized_ba, linearized_bg;
        Eigen::Matrix<double, 15, 15> jacobian, covariance, step_jacobian;
        Eigen::Matrix<double, 15, 18> step_V;
        double sum_dt;
        Eigen::Vector3d delta_p, delta_v;
        Eigen::Quaterniond delta_q;
        std::size_t sample_count;
        std::unique_ptr<equivariant::Preintegration> equivariant_preintegration;
        bool valid;
        std::uint64_t revision;
    };

    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : dt{0.0}, acc_0{_acc_0}, gyr_0{_gyr_0}, acc_1{_acc_0}, gyr_1{_gyr_0},
          linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
          covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          step_jacobian{Eigen::Matrix<double, 15, 15>::Zero()},
          step_V{Eigen::Matrix<double, 15, 18>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()},
          use_equivariant{IMU_PREINTEGRATION_ENABLE != 0}, valid{true}, revision_{0}

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

        if (use_equivariant)
        {
            equivariant_preintegration.reset(new equivariant::Preintegration(
                GYR_N, ACC_N, GYR_W, ACC_W,
                makeEquivariantBias(_linearized_ba, _linearized_bg)));
        }
    }

    std::unique_ptr<IntegrationBase> clone() const
    {
        std::unique_ptr<IntegrationBase> result(new IntegrationBase(
            linearized_acc, linearized_gyr, linearized_ba, linearized_bg));
        if (!result->commitFrom(*this))
            return nullptr;
        return result;
    }

    bool commitFrom(const IntegrationBase &source)
    {
        if (use_equivariant != source.use_equivariant ||
            !linearized_acc.isApprox(source.linearized_acc, 0.0) ||
            !linearized_gyr.isApprox(source.linearized_gyr, 0.0))
            return false;

        dt = source.dt;
        acc_0 = source.acc_0;
        gyr_0 = source.gyr_0;
        acc_1 = source.acc_1;
        gyr_1 = source.gyr_1;
        linearized_ba = source.linearized_ba;
        linearized_bg = source.linearized_bg;
        jacobian = source.jacobian;
        covariance = source.covariance;
        step_jacobian = source.step_jacobian;
        step_V = source.step_V;
        noise = source.noise;
        sum_dt = source.sum_dt;
        delta_p = source.delta_p;
        delta_q = source.delta_q;
        delta_v = source.delta_v;
        dt_buf = source.dt_buf;
        acc_buf = source.acc_buf;
        gyr_buf = source.gyr_buf;
        valid = source.valid;
        revision_ = source.revision_;
        if (source.equivariant_preintegration)
        {
            equivariant_preintegration.reset(new equivariant::Preintegration(
                *source.equivariant_preintegration));
        }
        else
        {
            equivariant_preintegration.reset();
        }
        return true;
    }

    Checkpoint checkpoint() const
    {
        Checkpoint result;
        result.dt = dt;
        result.acc_0 = acc_0;
        result.gyr_0 = gyr_0;
        result.acc_1 = acc_1;
        result.gyr_1 = gyr_1;
        result.linearized_ba = linearized_ba;
        result.linearized_bg = linearized_bg;
        result.jacobian = jacobian;
        result.covariance = covariance;
        result.step_jacobian = step_jacobian;
        result.step_V = step_V;
        result.sum_dt = sum_dt;
        result.delta_p = delta_p;
        result.delta_q = delta_q;
        result.delta_v = delta_v;
        result.sample_count = dt_buf.size();
        if (equivariant_preintegration)
        {
            result.equivariant_preintegration.reset(
                new equivariant::Preintegration(*equivariant_preintegration));
        }
        result.valid = valid;
        result.revision = revision_;
        return result;
    }

    void restore(Checkpoint &&source)
    {
        dt = source.dt;
        acc_0 = source.acc_0;
        gyr_0 = source.gyr_0;
        acc_1 = source.acc_1;
        gyr_1 = source.gyr_1;
        linearized_ba = source.linearized_ba;
        linearized_bg = source.linearized_bg;
        jacobian = source.jacobian;
        covariance = source.covariance;
        step_jacobian = source.step_jacobian;
        step_V = source.step_V;
        sum_dt = source.sum_dt;
        delta_p = source.delta_p;
        delta_q = source.delta_q;
        delta_v = source.delta_v;
        dt_buf.resize(source.sample_count);
        acc_buf.resize(source.sample_count);
        gyr_buf.resize(source.sample_count);
        equivariant_preintegration =
            std::move(source.equivariant_preintegration);
        valid = source.valid;
        revision_ = source.revision;
    }

    bool push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        if (!valid || !std::isfinite(dt) || dt <= 0.0 ||
            !acc.allFinite() || !gyr.allFinite())
        {
            ROS_WARN("Reject invalid IMU interval before preintegration");
            return false;
        }
        if (!propagate(dt, acc, gyr))
            return false;
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        ++revision_;
        return true;
    }

    bool repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        if (!_linearized_ba.allFinite() || !_linearized_bg.allFinite())
            return false;

        const double previous_dt = dt;
        const Eigen::Vector3d previous_acc_0 = acc_0;
        const Eigen::Vector3d previous_gyr_0 = gyr_0;
        const Eigen::Vector3d previous_acc_1 = acc_1;
        const Eigen::Vector3d previous_gyr_1 = gyr_1;
        const Eigen::Vector3d previous_ba = linearized_ba;
        const Eigen::Vector3d previous_bg = linearized_bg;
        const Eigen::Matrix<double, 15, 15> previous_jacobian = jacobian;
        const Eigen::Matrix<double, 15, 15> previous_covariance = covariance;
        const double previous_sum_dt = sum_dt;
        const Eigen::Vector3d previous_delta_p = delta_p;
        const Eigen::Quaterniond previous_delta_q = delta_q;
        const Eigen::Vector3d previous_delta_v = delta_v;
        const bool previous_valid = valid;
        std::unique_ptr<equivariant::Preintegration> previous_equivariant;
        if (equivariant_preintegration)
            previous_equivariant.reset(
                new equivariant::Preintegration(*equivariant_preintegration));

        sum_dt = 0.0;
        dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        acc_1 = linearized_acc;
        gyr_1 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        if (use_equivariant)
        {
            equivariant_preintegration->reset(
                makeEquivariantBias(_linearized_ba, _linearized_bg));
        }
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
        {
            if (propagate(dt_buf[i], acc_buf[i], gyr_buf[i]))
                continue;

            dt = previous_dt;
            acc_0 = previous_acc_0;
            gyr_0 = previous_gyr_0;
            acc_1 = previous_acc_1;
            gyr_1 = previous_gyr_1;
            linearized_ba = previous_ba;
            linearized_bg = previous_bg;
            jacobian = previous_jacobian;
            covariance = previous_covariance;
            sum_dt = previous_sum_dt;
            delta_p = previous_delta_p;
            delta_q = previous_delta_q;
            delta_v = previous_delta_v;
            valid = previous_valid;
            if (previous_equivariant)
                *equivariant_preintegration = *previous_equivariant;
            return false;
        }
        valid = previous_valid;
        ++revision_;
        return true;
    }

    bool equivariantIntegration(double integration_dt,
                                const Eigen::Vector3d &previous_acc,
                                const Eigen::Vector3d &previous_gyr,
                                const Eigen::Vector3d &current_acc,
                                const Eigen::Vector3d &current_gyr,
                                Eigen::Vector3d &result_delta_p,
                                Eigen::Quaterniond &result_delta_q,
                                Eigen::Vector3d &result_delta_v,
                                Eigen::Vector3d &result_linearized_ba,
                                Eigen::Vector3d &result_linearized_bg)
    {
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        equivariant::Preintegration candidate = *equivariant_preintegration;
        const bool integrated = candidate.integrate(
            previous_acc, previous_gyr, integration_dt);
        if (!integrated)
        {
            ROS_WARN("Equivariant IMU propagation produced invalid covariance");
            return false;
        }

        const equivariant::Gal3 &upsilon = candidate.upsilon();
        result_delta_q = Eigen::Quaterniond(upsilon.R()).normalized();
        result_delta_v = upsilon.v();
        result_delta_p = upsilon.p();
        Eigen::Matrix<double, 15, 15> result_covariance =
            convertEquivariantCovarianceToVins(candidate.covariance15());
        result_covariance = 0.5 *
            (result_covariance + result_covariance.transpose());
        if (!result_delta_q.coeffs().allFinite() ||
            !result_delta_v.allFinite() || !result_delta_p.allFinite() ||
            !result_covariance.allFinite())
            return false;

        *equivariant_preintegration = candidate;
        covariance = result_covariance;
        jacobian.setIdentity();
        return true;
    }

    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
            Eigen::Matrix<double, 18, 18> discrete_noise = noise / _dt;
            discrete_noise.topLeftCorner<12, 12>() *= 2.0;
            covariance = F * covariance * F.transpose() +
                         V * discrete_noise * V.transpose();
            covariance = 0.5 * (covariance + covariance.transpose());
        }

    }

    bool propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        if (!std::isfinite(_dt) || _dt <= 0.0 || !_acc_1.allFinite() ||
            !_gyr_1.allFinite())
            return false;

        Eigen::Matrix<double, 15, 15> previous_jacobian;
        Eigen::Matrix<double, 15, 15> previous_covariance;
        if (!use_equivariant)
        {
            previous_jacobian = jacobian;
            previous_covariance = covariance;
        }
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        if (use_equivariant)
        {
            if (!equivariantIntegration(
                    _dt, acc_0, gyr_0, _acc_1, _gyr_1,
                    result_delta_p, result_delta_q, result_delta_v,
                    result_linearized_ba, result_linearized_bg))
                return false;
        }
        else
        {
            midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                                linearized_ba, linearized_bg,
                                result_delta_p, result_delta_q, result_delta_v,
                                result_linearized_ba, result_linearized_bg, 1);
        }

        const double next_sum_dt = sum_dt + _dt;
        const double quaternion_norm = result_delta_q.norm();
        if (!result_delta_p.allFinite() || !result_delta_v.allFinite() ||
            !result_delta_q.coeffs().allFinite() ||
            !std::isfinite(quaternion_norm) || quaternion_norm <= 1e-12 ||
            !result_linearized_ba.allFinite() ||
            !result_linearized_bg.allFinite() || !jacobian.allFinite() ||
            !covariance.allFinite() || !std::isfinite(next_sum_dt))
        {
            if (!use_equivariant)
            {
                jacobian = previous_jacobian;
                covariance = previous_covariance;
            }
            return false;
        }

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        sum_dt = next_sum_dt;
        acc_0 = _acc_1;
        gyr_0 = _gyr_1;
        return true;
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        if (use_equivariant)
        {
            const Eigen::Vector3d predicted_p =
                Pi + Vi * sum_dt - 0.5 * G * sum_dt * sum_dt;
            const Eigen::Vector3d predicted_v = Vi - G * sum_dt;
            const equivariant::Preintegration::Vec10 bias_i =
                makeEquivariantBias(Bai, Bgi);
            const equivariant::Preintegration::Vec10 bias_j =
                makeEquivariantBias(Baj, Bgj);
            const equivariant::Gal3 corrected =
                equivariant_preintegration->correctedUpsilon(bias_i);
            const equivariant::Gal3 truth(
                Qi.inverse().toRotationMatrix() * Qj.toRotationMatrix(),
                Qi.inverse() * (Vj - predicted_v),
                Qi.inverse() * (Pj - predicted_p), sum_dt);
            const equivariant::Preintegration::Vec10 nav_error =
                equivariant::Gal3::log(truth * corrected.inverse());
            const equivariant::Preintegration::Vec10 bias_error =
                -equivariant::Gal3::inverseLeftJacobian(nav_error) *
                truth.adjoint() * (bias_j - bias_i);

            residuals.block<3, 1>(O_P, 0) = nav_error.segment<3>(6);
            residuals.block<3, 1>(O_R, 0) = nav_error.segment<3>(0);
            residuals.block<3, 1>(O_V, 0) = nav_error.segment<3>(3);
            residuals.block<3, 1>(O_BA, 0) = bias_error.segment<3>(3);
            residuals.block<3, 1>(O_BG, 0) = bias_error.segment<3>(0);
            return residuals;
        }

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    bool usesEquivariantPreintegration() const
    {
        return use_equivariant;
    }

    bool isValidForFactor() const
    {
        return valid && sum_dt > 0.0 && std::isfinite(sum_dt) &&
               delta_p.allFinite() && delta_q.coeffs().allFinite() &&
               delta_v.allFinite() && covariance.allFinite();
    }

    void markInvalid()
    {
        if (valid)
        {
            valid = false;
            ++revision_;
        }
    }

    bool isValid() const { return valid; }

    std::uint64_t revision() const
    {
        return revision_;
    }

    Eigen::Quaterniond correctedDeltaQ(const Eigen::Vector3d &ba,
                                       const Eigen::Vector3d &bg) const
    {
        if (use_equivariant)
        {
            return Eigen::Quaterniond(
                equivariant_preintegration->correctedUpsilon(
                    makeEquivariantBias(ba, bg)).R()).normalized();
        }

        const Eigen::Vector3d dbg = bg - linearized_bg;
        const Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
        return (delta_q * Utility::deltaQ(dq_dbg * dbg)).normalized();
    }

    static equivariant::Preintegration::Vec10 makeEquivariantBias(
        const Eigen::Vector3d &ba, const Eigen::Vector3d &bg)
    {
        equivariant::Preintegration::Vec10 bias =
            equivariant::Preintegration::Vec10::Zero();
        bias.segment<3>(0) = bg;
        bias.segment<3>(3) = ba;
        return bias;
    }

    static Eigen::Matrix<double, 15, 15> convertEquivariantCovarianceToVins(
        const equivariant::Preintegration::Mat15 &equivariant_covariance)
    {
        const int equivariant_offsets[5] = {6, 0, 3, 12, 9};
        const int vins_offsets[5] = {O_P, O_R, O_V, O_BA, O_BG};
        Eigen::Matrix<double, 15, 15> vins_covariance;
        for (int row = 0; row < 5; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                vins_covariance.block<3, 3>(vins_offsets[row], vins_offsets[column]) =
                    equivariant_covariance.block<3, 3>(equivariant_offsets[row],
                                                       equivariant_offsets[column]);
            }
        }
        return vins_covariance;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

    const bool use_equivariant;
    std::unique_ptr<equivariant::Preintegration> equivariant_preintegration;
    bool valid;
    std::uint64_t revision_;

};
/*

    void eulerIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }     


    void checkJacobian(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0, 
                                   const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb p       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb q       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb v       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb ba       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb bg       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << endl;
    }
    */
