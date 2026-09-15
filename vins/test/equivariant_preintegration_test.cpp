#include "IMU_core/equivariant_preintegration.h"

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace
{
using equivariant::Gal3;
using equivariant::Preintegration;

bool check(bool condition, const std::string &message)
{
    if (!condition)
        std::cerr << "FAILED: " << message << std::endl;
    return condition;
}

Eigen::Matrix<double, 5, 5> matrix(const Gal3 &X)
{
    Eigen::Matrix<double, 5, 5> result = Eigen::Matrix<double, 5, 5>::Identity();
    result.block<3, 3>(0, 0) = X.R();
    result.block<3, 1>(0, 3) = X.v();
    result.block<3, 1>(0, 4) = X.p();
    result(3, 4) = X.s();
    return result;
}

bool testGroupOperations()
{
    Gal3::Vec10 x;
    x << 0.2, -0.1, 0.05, 1.0, -0.3, 0.2, 0.4, 0.1, -0.2, 0.03;
    Gal3::Vec10 y;
    y << -0.1, 0.04, 0.08, -0.2, 0.5, 0.1, 0.3, -0.4, 0.2, -0.01;
    const Gal3 X = Gal3::exp(x);
    const Gal3 Y = Gal3::exp(y);

    Eigen::Matrix<double, 5, 5> algebra = Eigen::Matrix<double, 5, 5>::Zero();
    algebra.block<3, 3>(0, 0) = Gal3::skew(x.segment<3>(0));
    algebra.block<3, 1>(0, 3) = x.segment<3>(3);
    algebra.block<3, 1>(0, 4) = x.segment<3>(6);
    algebra(3, 4) = x(9);

    bool ok = true;
    ok &= check((Gal3::log(X) - x).norm() < 1e-11, "Gal3 Exp/Log round trip");
    ok &= check((matrix(X) - algebra.exp()).norm() < 1e-12,
                "Gal3 closed-form exponential");
    ok &= check((matrix(X * Y) - matrix(X) * matrix(Y)).norm() < 1e-12,
                "Gal3 product matches homogeneous matrices");
    ok &= check((matrix(X.inverse()) * matrix(X) -
                 Eigen::Matrix<double, 5, 5>::Identity()).norm() < 1e-12,
                "Gal3 inverse");

    const double eps = 1e-7;
    const Gal3 conjugated = X * Gal3::exp(eps * y) * X.inverse();
    ok &= check((Gal3::log(conjugated) / eps - X.adjoint() * y).norm() < 1e-7,
                "Gal3 adjoint");
    return ok;
}

bool testLeftJacobian()
{
    Gal3::Vec10 x;
    x << 0.3, -0.2, 0.1, 0.8, -0.4, 0.3, 0.2, 0.1, -0.5, 0.02;
    const Gal3::Mat10 J = Gal3::leftJacobian(x);
    const Gal3 X_inverse = Gal3::exp(x).inverse();
    const double eps = 1e-7;
    Gal3::Mat10 numerical;
    for (int axis = 0; axis < 10; ++axis)
    {
        Gal3::Vec10 perturbation = Gal3::Vec10::Zero();
        perturbation(axis) = eps;
        const Gal3 plus = Gal3::exp(x + perturbation) * X_inverse;
        perturbation(axis) = -eps;
        const Gal3 minus = Gal3::exp(x + perturbation) * X_inverse;
        numerical.col(axis) = (Gal3::log(plus) - Gal3::log(minus)) / (2.0 * eps);
    }

    Gal3::Vec10 zero_rotation = Gal3::Vec10::Zero();
    zero_rotation.segment<3>(3) << 0.7, -0.2, 0.4;
    zero_rotation(9) = 0.03;
    const Gal3::Mat10 J_zero = Gal3::leftJacobian(zero_rotation);
    Gal3::Vec10 direction;
    direction << -0.2, 0.3, 0.1, 0.4, -0.1, 0.2, -0.3, 0.5, 0.2, -0.01;
    const Gal3::Mat10 directional =
        Gal3::leftJacobianDirectionalDerivative(x, direction);
    const Gal3::Mat10 numerical_directional =
        (Gal3::leftJacobian(x + eps * direction) -
         Gal3::leftJacobian(x - eps * direction)) / (2.0 * eps);
    Gal3::Vec10 transported;
    transported << 0.1, -0.2, 0.05, 0.3, 0.1, -0.2, 0.4, -0.3, 0.2, 0.0;
    const Gal3::Mat10 inverse_left = Gal3::inverseLeftJacobian(x);
    const Gal3::Vec10 coupled = -inverse_left * transported;
    const Gal3::Vec10 coupled_directional =
        -inverse_left * directional * coupled;
    const Gal3::Vec10 coupled_plus =
        -Gal3::inverseLeftJacobian(x + eps * direction) * transported;
    const Gal3::Vec10 coupled_minus =
        -Gal3::inverseLeftJacobian(x - eps * direction) * transported;
    const Gal3::Vec10 numerical_coupled_directional =
        (coupled_plus - coupled_minus) / (2.0 * eps);

    const Gal3 T = Gal3::exp(x);
    const Gal3::Vec10 adjoint_directional =
        Gal3::algebraAdjoint(direction) * T.adjoint() * transported;
    const Gal3::Vec10 numerical_adjoint_directional =
        ((Gal3::exp(eps * direction) * T).adjoint() * transported -
         (Gal3::exp(-eps * direction) * T).adjoint() * transported) /
        (2.0 * eps);

    const Gal3 corrected = Gal3::exp(0.5 * direction);
    const auto coupledBias = [&](const Gal3 &truth) -> Gal3::Vec10
    {
        const Gal3::Vec10 error = Gal3::log(truth * corrected.inverse());
        const Gal3::Vec10 value = -Gal3::inverseLeftJacobian(error) *
                                  truth.adjoint() * transported;
        return value;
    };
    const Gal3 error = T * corrected.inverse();
    const Gal3::Vec10 error_coordinates = Gal3::log(error);
    const Gal3::Mat10 error_inverse_left =
        Gal3::inverseLeftJacobian(error_coordinates);
    const Gal3::Vec10 base_coupled_bias = coupledBias(T);
    const Gal3::Vec10 error_direction =
        error_inverse_left * direction;
    const Gal3::Vec10 numerical_error_direction =
        (Gal3::log(Gal3::exp(eps * direction) * error) -
         Gal3::log(Gal3::exp(-eps * direction) * error)) / (2.0 * eps);
    Gal3::Mat10 coupled_log_jacobian;
    for (int axis = 0; axis < 10; ++axis)
    {
        Gal3::Vec10 basis = Gal3::Vec10::Zero();
        basis(axis) = 1.0;
        coupled_log_jacobian.col(axis) =
            -error_inverse_left *
            Gal3::leftJacobianDirectionalDerivative(error_coordinates, basis) *
            base_coupled_bias;
    }
    const Gal3::Vec10 complete_coupled_directional =
        coupled_log_jacobian * error_direction -
        error_inverse_left * Gal3::algebraAdjoint(direction) *
            T.adjoint() * transported;
    const Gal3::Vec10 numerical_complete_coupled_directional =
        (coupledBias(Gal3::exp(eps * direction) * T) -
         coupledBias(Gal3::exp(-eps * direction) * T)) / (2.0 * eps);
    const Eigen::Matrix3d expected_position_gyro =
        -zero_rotation(9) / 6.0 * Gal3::skew(zero_rotation.segment<3>(3));

    bool ok = true;
    ok &= check((J - numerical).norm() < 2e-7, "complete Gal3 left Jacobian");
    ok &= check((J * Gal3::inverseLeftJacobian(x) - Gal3::Mat10::Identity()).norm() < 1e-11,
                "inverse Gal3 left Jacobian");
    ok &= check((J_zero.block<3, 3>(6, 0) - expected_position_gyro).norm() < 1e-14,
                "zero-rate Gal3 Q2 term");
    ok &= check((directional - numerical_directional).norm() < 2e-8,
                "Gal3 left Jacobian directional derivative");
    ok &= check((coupled_directional - numerical_coupled_directional).norm() < 2e-8,
                "inverse left Jacobian directional derivative");
    ok &= check((adjoint_directional - numerical_adjoint_directional).norm() < 2e-8,
                "Gal3 adjoint left-perturbation derivative");
    const double complete_coupled_error =
        (complete_coupled_directional -
         numerical_complete_coupled_directional).norm();
    if (complete_coupled_error >= 2e-8)
        std::cerr << "complete coupled derivative error: "
                  << complete_coupled_error
                  << " analytic_norm=" << complete_coupled_directional.norm()
                  << " numerical_norm=" << numerical_complete_coupled_directional.norm()
                  << " error_direction_norm=" << error_direction.norm()
                  << " H_norm=" << coupled_log_jacobian.norm()
                  << std::endl;
    ok &= check(complete_coupled_error < 2e-8,
                "complete coupled bias directional derivative");
    ok &= check((error_direction - numerical_error_direction).norm() < 2e-8,
                "Gal3 Log left-perturbation derivative");
    return ok;
}

bool testPreintegration()
{
    const Preintegration::Vec10 bias = Preintegration::Vec10::Zero();
    Preintegration split(0.01, 0.1, 0.0001, 0.001, bias);
    Preintegration single(0.01, 0.1, 0.0001, 0.001, bias);
    const Eigen::Vector3d acc(0.4, -0.2, 9.7);
    const Eigen::Vector3d gyro(0.1, -0.05, 0.02);
    for (int i = 0; i < 100; ++i)
        if (!split.integrate(acc, gyro, 0.01))
            return check(false, "split preintegration accepted finite positive dt");
    if (!single.integrate(acc, gyro, 1.0))
        return check(false, "single preintegration accepted finite positive dt");

    bool ok = true;
    ok &= check(Gal3::log(split.upsilon() * single.upsilon().inverse()).norm() < 1e-10,
                "constant-input preintegration composes exactly");
    ok &= check(!split.integrate(acc, gyro, 0.0), "zero dt rejected");
    ok &= check(split.covariance15().allFinite(), "preintegration covariance finite");
    Eigen::SelfAdjointEigenSolver<Preintegration::Mat15> eigensolver(split.covariance15());
    ok &= check(eigensolver.info() == Eigen::Success &&
                eigensolver.eigenvalues().minCoeff() > -1e-12,
                "preintegration covariance positive semidefinite");

    const double step_dt = 0.01;
    const double gyro_density = 0.02;
    const double acc_density = 0.2;
    const double gyro_walk = 0.002;
    const double acc_walk = 0.02;
    Preintegration one_step(gyro_density, acc_density, gyro_walk, acc_walk, bias);
    ok &= check(one_step.integrate(Eigen::Vector3d::Zero(),
                                   Eigen::Vector3d::Zero(), step_dt),
                "one-step covariance propagation");
    const Preintegration::Mat15 one_step_covariance = one_step.covariance15();
    ok &= check((one_step_covariance.block<3, 3>(0, 0) -
                 gyro_density * gyro_density * step_dt *
                     Eigen::Matrix3d::Identity()).norm() < 1e-12,
                "gyro density has continuous-time dt scaling");
    ok &= check((one_step_covariance.block<3, 3>(3, 3) -
                 acc_density * acc_density * step_dt *
                     Eigen::Matrix3d::Identity()).norm() < 1e-12,
                "accelerometer density has continuous-time dt scaling");
    ok &= check((one_step_covariance.block<3, 3>(9, 9) -
                 gyro_walk * gyro_walk * step_dt *
                     Eigen::Matrix3d::Identity()).norm() < 1e-12,
                "gyro bias random walk has continuous-time dt scaling");
    ok &= check((one_step_covariance.block<3, 3>(12, 12) -
                 acc_walk * acc_walk * step_dt *
                     Eigen::Matrix3d::Identity()).norm() < 1e-12,
                "accelerometer bias random walk has continuous-time dt scaling");

    Preintegration::Vec10 perturbed_bias = bias;
    perturbed_bias.segment<3>(0) << 1e-5, -2e-5, 1.5e-5;
    perturbed_bias.segment<3>(3) << -1e-5, 0.5e-5, 2e-5;
    Preintegration exact(0.01, 0.1, 0.0001, 0.001, perturbed_bias);
    for (int i = 0; i < 100; ++i)
        exact.integrate(acc, gyro, 0.01);
    const Gal3 corrected = split.correctedUpsilon(perturbed_bias);
    ok &= check(Gal3::log(exact.upsilon() * corrected.inverse()).norm() < 1e-7,
                "first-order equivariant bias correction");

    Preintegration transactional(gyro_density, acc_density, gyro_walk, acc_walk, bias);
    const Gal3 before_invalid = transactional.upsilon();
    const Preintegration::Mat15 covariance_before_invalid =
        transactional.covariance15();
    ok &= check(!transactional.integrate(
                    acc, gyro, std::numeric_limits<double>::infinity()),
                "non-finite dt rejected");
    ok &= check(
        Gal3::log(transactional.upsilon() * before_invalid.inverse()).norm() == 0.0 &&
            (transactional.covariance15() - covariance_before_invalid).norm() == 0.0,
        "rejected propagation leaves state unchanged");
    return ok;
}
} // namespace

int main()
{
    const bool ok = testGroupOperations() && testLeftJacobian() && testPreintegration();
    if (ok)
        std::cout << "Equivariant preintegration tests passed" << std::endl;
    return ok ? 0 : 1;
}
