/*******************************************************
 * Copyright (C) 2026 VINS-Fusion contributors
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0.
 *
 * Independent implementation of the Gal(3) preintegration equations in:
 * G. Delama et al., "Equivariant IMU Preintegration With Biases:
 * A Galilean Group Approach," IEEE RA-L, 2025.
 *
 * Version history: v1.0 introduced Gal(3) equivariant preintegration with
 * numerical factor Jacobians; v1.1 replaced those Jacobians analytically;
 * v1.2 keeps the model and unifies its ZOH propagation and noise semantics.
 *******************************************************/

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace equivariant
{

class Gal3
{
  public:
    using Vec10 = Eigen::Matrix<double, 10, 1>;
    using Mat10 = Eigen::Matrix<double, 10, 10>;

    Gal3()
        : R_(Eigen::Matrix3d::Identity()), v_(Eigen::Vector3d::Zero()),
          p_(Eigen::Vector3d::Zero()), s_(0.0)
    {
    }

    Gal3(const Eigen::Matrix3d &R, const Eigen::Vector3d &v,
         const Eigen::Vector3d &p, double s)
        : R_(R), v_(v), p_(p), s_(s)
    {
    }

    const Eigen::Matrix3d &R() const { return R_; }
    const Eigen::Vector3d &v() const { return v_; }
    const Eigen::Vector3d &p() const { return p_; }
    double s() const { return s_; }

    Gal3 operator*(const Gal3 &other) const
    {
        return Gal3(R_ * other.R_, v_ + R_ * other.v_,
                    p_ + R_ * other.p_ + v_ * other.s_, s_ + other.s_);
    }

    Gal3 inverse() const
    {
        const Eigen::Matrix3d Rt = R_.transpose();
        return Gal3(Rt, -Rt * v_, Rt * (v_ * s_ - p_), -s_);
    }

    Mat10 adjoint() const
    {
        Mat10 Ad = Mat10::Zero();
        Ad.block<3, 3>(0, 0) = R_;
        Ad.block<3, 3>(3, 0) = skew(v_) * R_;
        Ad.block<3, 3>(3, 3) = R_;
        Ad.block<3, 3>(6, 0) = skew(p_ - v_ * s_) * R_;
        Ad.block<3, 3>(6, 3) = -s_ * R_;
        Ad.block<3, 3>(6, 6) = R_;
        Ad.block<3, 1>(6, 9) = v_;
        Ad(9, 9) = 1.0;
        return Ad;
    }

    static Gal3 exp(const Vec10 &x)
    {
        const Eigen::Vector3d omega = x.segment<3>(0);
        const Eigen::Vector3d nu = x.segment<3>(3);
        const Eigen::Vector3d rho = x.segment<3>(6);
        const double sigma = x(9);
        const Eigen::Matrix3d J = gamma1(omega);
        return Gal3(rotationExp(omega), J * nu,
                    J * rho + sigma * gamma2(omega) * nu, sigma);
    }

    static Vec10 log(const Gal3 &X)
    {
        Vec10 x;
        const Eigen::Vector3d omega = rotationLog(X.R_);
        const Eigen::Matrix3d J_inv = gamma1Inverse(omega);
        const Eigen::Vector3d nu = J_inv * X.v_;
        x.segment<3>(0) = omega;
        x.segment<3>(3) = nu;
        x.segment<3>(6) = J_inv * (X.p_ - X.s_ * gamma2(omega) * nu);
        x(9) = X.s_;
        return x;
    }

    static Mat10 algebraAdjoint(const Vec10 &x)
    {
        Mat10 ad = Mat10::Zero();
        const Eigen::Matrix3d W = skew(x.segment<3>(0));
        ad.block<3, 3>(0, 0) = W;
        ad.block<3, 3>(3, 0) = skew(x.segment<3>(3));
        ad.block<3, 3>(3, 3) = W;
        ad.block<3, 3>(6, 0) = skew(x.segment<3>(6));
        ad.block<3, 3>(6, 3) = -x(9) * Eigen::Matrix3d::Identity();
        ad.block<3, 3>(6, 6) = W;
        ad.block<3, 1>(6, 9) = x.segment<3>(3);
        return ad;
    }

    static Mat10 leftJacobian(const Vec10 &x)
    {
        const Mat10 ad = algebraAdjoint(x);
        Mat10 result = Mat10::Identity();
        Mat10 term = Mat10::Identity();
        for (int order = 1; order <= 80; ++order)
        {
            term = (term * ad) / static_cast<double>(order + 1);
            result += term;
            if (term.cwiseAbs().maxCoeff() < 1e-15)
                break;
        }
        return result;
    }

    static Mat10 inverseLeftJacobian(const Vec10 &x)
    {
        return leftJacobian(x).partialPivLu().solve(Mat10::Identity());
    }

    static Mat10 leftJacobianDirectionalDerivative(
        const Vec10 &x, const Vec10 &direction)
    {
        const Mat10 ad = algebraAdjoint(x);
        const Mat10 direction_ad = algebraAdjoint(direction);
        Mat10 term = Mat10::Identity();
        Mat10 derivative = Mat10::Zero();
        Mat10 result = Mat10::Zero();
        for (int order = 1; order <= 80; ++order)
        {
            const double denominator = static_cast<double>(order + 1);
            const Mat10 next_derivative =
                (derivative * ad + term * direction_ad) / denominator;
            const Mat10 next_term = (term * ad) / denominator;
            result += next_derivative;
            derivative = next_derivative;
            term = next_term;
            if (term.cwiseAbs().maxCoeff() < 1e-15 &&
                derivative.cwiseAbs().maxCoeff() < 1e-15)
                break;
        }
        return result;
    }

    static Eigen::Matrix3d skew(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d result;
        result << 0.0, -v.z(), v.y(),
                  v.z(), 0.0, -v.x(),
                  -v.y(), v.x(), 0.0;
        return result;
    }

  private:
    static Eigen::Matrix3d rotationExp(const Eigen::Vector3d &omega)
    {
        const double theta2 = omega.squaredNorm();
        const Eigen::Matrix3d W = skew(omega);
        double a;
        double b;
        if (theta2 < 1e-12)
        {
            const double theta4 = theta2 * theta2;
            a = 1.0 - theta2 / 6.0 + theta4 / 120.0;
            b = 0.5 - theta2 / 24.0 + theta4 / 720.0;
        }
        else
        {
            const double theta = std::sqrt(theta2);
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / theta2;
        }
        return Eigen::Matrix3d::Identity() + a * W + b * W * W;
    }

    static Eigen::Vector3d rotationLog(const Eigen::Matrix3d &R)
    {
        Eigen::Quaterniond q(R);
        q.normalize();
        if (q.w() < 0.0)
            q.coeffs() *= -1.0;
        const double sin_half_theta = q.vec().norm();
        if (sin_half_theta < 1e-12)
            return 2.0 * q.vec();
        const double theta = 2.0 * std::atan2(sin_half_theta, q.w());
        return theta * q.vec() / sin_half_theta;
    }

    static Eigen::Matrix3d gamma1(const Eigen::Vector3d &omega)
    {
        const double theta2 = omega.squaredNorm();
        const Eigen::Matrix3d W = skew(omega);
        double b;
        double c;
        if (theta2 < 1e-12)
        {
            const double theta4 = theta2 * theta2;
            b = 0.5 - theta2 / 24.0 + theta4 / 720.0;
            c = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0;
        }
        else
        {
            const double theta = std::sqrt(theta2);
            b = (1.0 - std::cos(theta)) / theta2;
            c = (theta - std::sin(theta)) / (theta2 * theta);
        }
        return Eigen::Matrix3d::Identity() + b * W + c * W * W;
    }

    static Eigen::Matrix3d gamma2(const Eigen::Vector3d &omega)
    {
        const double theta2 = omega.squaredNorm();
        const Eigen::Matrix3d W = skew(omega);
        double c;
        double d;
        if (theta2 < 1e-12)
        {
            const double theta4 = theta2 * theta2;
            c = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0;
            d = 1.0 / 24.0 - theta2 / 720.0 + theta4 / 40320.0;
        }
        else
        {
            const double theta = std::sqrt(theta2);
            c = (theta - std::sin(theta)) / (theta2 * theta);
            d = (theta2 + 2.0 * std::cos(theta) - 2.0) /
                (2.0 * theta2 * theta2);
        }
        return 0.5 * Eigen::Matrix3d::Identity() + c * W + d * W * W;
    }

    static Eigen::Matrix3d gamma1Inverse(const Eigen::Vector3d &omega)
    {
        const double theta2 = omega.squaredNorm();
        const Eigen::Matrix3d W = skew(omega);
        double coefficient;
        if (theta2 < 1e-12)
        {
            coefficient = 1.0 / 12.0 + theta2 / 720.0 +
                          theta2 * theta2 / 30240.0;
        }
        else
        {
            const double theta = std::sqrt(theta2);
            coefficient = 1.0 / theta2 -
                          (1.0 + std::cos(theta)) /
                              (2.0 * theta * std::sin(theta));
        }
        return Eigen::Matrix3d::Identity() - 0.5 * W + coefficient * W * W;
    }

    Eigen::Matrix3d R_;
    Eigen::Vector3d v_;
    Eigen::Vector3d p_;
    double s_;
};

inline bool propagateWorldStateZoh(
    double dt, const Eigen::Vector3d &left_acc,
    const Eigen::Vector3d &left_gyro, const Eigen::Vector3d &ba,
    const Eigen::Vector3d &bg, const Eigen::Vector3d &gravity,
    Eigen::Matrix3d &R, Eigen::Vector3d &V, Eigen::Vector3d &P)
{
    if (!std::isfinite(dt) || dt <= 0.0 || !left_acc.allFinite() ||
        !left_gyro.allFinite() || !ba.allFinite() || !bg.allFinite() ||
        !gravity.allFinite() || !R.allFinite() || !V.allFinite() ||
        !P.allFinite())
        return false;

    Gal3::Vec10 input = Gal3::Vec10::Zero();
    input.segment<3>(0) = left_gyro - bg;
    input.segment<3>(3) = left_acc - ba;
    input(9) = 1.0;
    const Gal3 delta = Gal3::exp(input * dt);
    const Eigen::Matrix3d R0 = R;
    const Eigen::Vector3d V0 = V;
    const Eigen::Vector3d P0 = P;
    const Eigen::Matrix3d next_R = R0 * delta.R();
    const Eigen::Vector3d next_V = V0 + R0 * delta.v() - gravity * dt;
    const Eigen::Vector3d next_P = P0 + V0 * dt + R0 * delta.p() -
                                   0.5 * gravity * dt * dt;
    if (!next_R.allFinite() || !next_V.allFinite() || !next_P.allFinite())
        return false;

    R = next_R;
    V = next_V;
    P = next_P;
    return true;
}

class Preintegration
{
  public:
    using Vec10 = Gal3::Vec10;
    using Mat10 = Gal3::Mat10;
    using Mat15 = Eigen::Matrix<double, 15, 15>;
    using Mat20 = Eigen::Matrix<double, 20, 20>;

    Preintegration(double gyro_noise_density, double acc_noise_density,
                   double gyro_bias_random_walk, double acc_bias_random_walk,
                   const Vec10 &bias_hat)
        : bias_hat_(bias_hat)
    {
        continuous_noise_.setZero();
        continuous_noise_.block<3, 3>(0, 0) =
            gyro_noise_density * gyro_noise_density * Eigen::Matrix3d::Identity();
        continuous_noise_.block<3, 3>(3, 3) =
            acc_noise_density * acc_noise_density * Eigen::Matrix3d::Identity();
        continuous_noise_.block<3, 3>(10, 10) =
            gyro_bias_random_walk * gyro_bias_random_walk * Eigen::Matrix3d::Identity();
        continuous_noise_.block<3, 3>(13, 13) =
            acc_bias_random_walk * acc_bias_random_walk * Eigen::Matrix3d::Identity();
        reset(bias_hat);
    }

    void reset(const Vec10 &bias_hat)
    {
        bias_hat_ = bias_hat;
        upsilon_ = Gal3();
        covariance_.setZero();
        bias_jacobian_.setIdentity();
    }

    bool integrate(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro,
                   double dt)
    {
        if (!std::isfinite(dt) || dt <= 0.0 || !acc.allFinite() || !gyro.allFinite())
            return false;

        Vec10 measurement = Vec10::Zero();
        measurement.segment<3>(0) = gyro;
        measurement.segment<3>(3) = acc;
        measurement(9) = 1.0;

        const Vec10 unbiased = measurement - bias_hat_;
        const Vec10 increment = unbiased * dt;
        const Mat10 K = upsilon_.adjoint() * Gal3::leftJacobian(increment) * dt;
        const Vec10 origin_input = upsilon_.adjoint() * unbiased;
        const Gal3 next_upsilon = upsilon_ * Gal3::exp(increment);

        Mat20 A = Mat20::Identity();
        A.block<10, 10>(0, 10) = Gal3::leftJacobian(origin_input * dt) * dt;
        A.block<10, 10>(10, 10) = Gal3::exp(origin_input * dt).adjoint();

        Mat20 B = Mat20::Zero();
        B.block<10, 10>(0, 0) = -K;
        B.block<10, 10>(10, 10) = next_upsilon.adjoint() * dt;

        Mat20 next_covariance = A * covariance_ * A.transpose() +
                                B * (continuous_noise_ / dt) * B.transpose();
        next_covariance = 0.5 * (next_covariance + next_covariance.transpose());

        Mat20 Phi_bias = Mat20::Identity();
        Phi_bias.block<10, 10>(0, 10) = -K;
        const Mat20 next_bias_jacobian = Phi_bias * bias_jacobian_;
        if (!next_covariance.allFinite() || !next_bias_jacobian.allFinite() ||
            !next_upsilon.R().allFinite() || !next_upsilon.v().allFinite() ||
            !next_upsilon.p().allFinite() || !std::isfinite(next_upsilon.s()))
            return false;

        covariance_ = next_covariance;
        bias_jacobian_ = next_bias_jacobian;
        upsilon_ = next_upsilon;
        return true;
    }

    const Gal3 &upsilon() const { return upsilon_; }
    const Vec10 &biasHat() const { return bias_hat_; }
    Mat10 biasCorrectionJacobian() const
    {
        return bias_jacobian_.block<10, 10>(0, 10);
    }

    Gal3 correctedUpsilon(const Vec10 &bias) const
    {
        return Gal3::exp(biasCorrectionJacobian() * (bias - bias_hat_)) * upsilon_;
    }

    Mat15 covariance15() const
    {
        Mat15 result;
        result.block<9, 9>(0, 0) = covariance_.block<9, 9>(0, 0);
        result.block<9, 6>(0, 9) = covariance_.block<9, 6>(0, 10);
        result.block<6, 9>(9, 0) = covariance_.block<6, 9>(10, 0);
        result.block<6, 6>(9, 9) = covariance_.block<6, 6>(10, 10);
        return result;
    }

  private:
    Vec10 bias_hat_;
    Gal3 upsilon_;
    Mat20 continuous_noise_;
    Mat20 covariance_;
    Mat20 bias_jacobian_;
};

} // namespace equivariant
