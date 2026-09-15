#include "factor/imu_factor.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
bool check(bool condition, const std::string &message)
{
    if (!condition)
        std::cerr << "FAILED: " << message << std::endl;
    return condition;
}

void setPose(double *parameters, const Eigen::Vector3d &position,
             const Eigen::Quaterniond &orientation)
{
    parameters[0] = position.x();
    parameters[1] = position.y();
    parameters[2] = position.z();
    parameters[3] = orientation.x();
    parameters[4] = orientation.y();
    parameters[5] = orientation.z();
    parameters[6] = orientation.w();
}

void setSpeedBias(double *parameters, const Eigen::Vector3d &velocity,
                  const Eigen::Vector3d &ba, const Eigen::Vector3d &bg)
{
    Eigen::Map<Eigen::Vector3d> velocity_map(parameters);
    Eigen::Map<Eigen::Vector3d> ba_map(parameters + 3);
    Eigen::Map<Eigen::Vector3d> bg_map(parameters + 6);
    velocity_map = velocity;
    ba_map = ba;
    bg_map = bg;
}

using FactorState = std::array<std::vector<double>, 4>;

Eigen::Matrix<double, 15, 1> evaluateFactor(IMUFactor &factor,
                                             const FactorState &state)
{
    const double *parameters[4] = {
        state[0].data(), state[1].data(), state[2].data(), state[3].data()};
    Eigen::Matrix<double, 15, 1> residual;
    if (!factor.Evaluate(parameters, residual.data(), nullptr))
        residual.setConstant(std::numeric_limits<double>::quiet_NaN());
    return residual;
}

void perturbState(FactorState &state, int block, int column, double delta)
{
    if ((block == 0 || block == 2) && column >= 3)
    {
        Eigen::Quaterniond q(state[block][6], state[block][3],
                             state[block][4], state[block][5]);
        Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
        dtheta(column - 3) = delta;
        q = (q * Utility::deltaQ(dtheta)).normalized();
        state[block][3] = q.x();
        state[block][4] = q.y();
        state[block][5] = q.z();
        state[block][6] = q.w();
    }
    else
    {
        state[block][column] += delta;
    }
}

bool checkJacobianColumn(IMUFactor &factor, const FactorState &base,
                         int block, int column,
                         const Eigen::Matrix<double, 15, 1> &implemented)
{
    const double epsilon = 2e-5;
    FactorState plus = base;
    FactorState minus = base;
    perturbState(plus, block, column, epsilon);
    perturbState(minus, block, column, -epsilon);
    const Eigen::Matrix<double, 15, 1> numerical =
        (evaluateFactor(factor, plus) - evaluateFactor(factor, minus)) /
        (2.0 * epsilon);
    const double relative_error = (implemented - numerical).norm() /
        std::max(1.0, numerical.norm());
    constexpr double max_relative_error = 2e-5;
    if (!numerical.allFinite() || relative_error >= max_relative_error)
    {
        std::cerr << "Jacobian mismatch block=" << block
                  << " column=" << column
                  << " relative_error=" << relative_error
                  << " implemented_norm=" << implemented.norm()
                  << " numerical_norm=" << numerical.norm()
                  << " nav_error=" << (implemented.head<9>() - numerical.head<9>()).norm()
                  << " bias_error=" << (implemented.tail<6>() - numerical.tail<6>()).norm()
                  << std::endl;
        if (block == 0 && column == 0)
            std::cerr << "implemented=" << implemented.transpose()
                      << "\nnumerical=" << numerical.transpose() << std::endl;
    }
    return check(numerical.allFinite() && relative_error < max_relative_error,
                  "equivariant factor Jacobian agrees with independent step");
}

bool testMode(int mode)
{
    IMU_PREINTEGRATION_ENABLE = mode;
    ACC_N = 0.1;
    GYR_N = 0.01;
    ACC_W = 0.001;
    GYR_W = 0.0001;
    EQUIVARIANT_ACC_N = 0.1;
    EQUIVARIANT_GYR_N = 0.01;
    EQUIVARIANT_ACC_W = 0.001;
    EQUIVARIANT_GYR_W = 0.0001;
    G = Eigen::Vector3d(0.0, 0.0, 9.805);

    const Eigen::Vector3d ba(0.01, -0.02, 0.005);
    const Eigen::Vector3d bg(0.001, -0.002, 0.0005);
    const Eigen::Vector3d acc(0.3, -0.1, 9.75);
    const Eigen::Vector3d gyro(0.08, -0.04, 0.02);
    IntegrationBase preintegration(acc, gyro, ba, bg);
    for (int i = 0; i < 100; ++i)
        preintegration.push_back(0.01, acc, gyro);

    const Eigen::Vector3d Pi(1.0, -0.5, 0.2);
    const Eigen::Quaterniond Qi =
        Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX());
    const Eigen::Vector3d Vi(0.2, 0.1, -0.05);
    const double dt = preintegration.sum_dt;
    const Eigen::Quaterniond Qj = Qi * preintegration.delta_q;
    const Eigen::Vector3d Vj = Vi - G * dt + Qi * preintegration.delta_v;
    const Eigen::Vector3d Pj = Pi + Vi * dt - 0.5 * G * dt * dt +
                               Qi * preintegration.delta_p;

    bool ok = true;
    ok &= check(preintegration.usesEquivariantPreintegration() == (mode != 0),
                "preintegration mode snapshot");
    ok &= check(preintegration.evaluate(Pi, Qi, Vi, ba, bg, Pj, Qj, Vj, ba, bg).norm() < 1e-9,
                mode ? "zero equivariant residual" : "zero classic residual");

    if (mode)
    {
        const Eigen::Vector3d next_acc = acc + Eigen::Vector3d(4.0, -3.0, 2.0);
        const Eigen::Vector3d next_gyro = gyro + Eigen::Vector3d(1.0, 0.5, -0.7);
        IntegrationBase zoh(acc, gyro, ba, bg);
        zoh.push_back(0.02, next_acc, next_gyro);
        equivariant::Preintegration::Vec10 increment =
            equivariant::Preintegration::Vec10::Zero();
        increment.segment<3>(0) = gyro - bg;
        increment.segment<3>(3) = acc - ba;
        increment(9) = 1.0;
        const equivariant::Gal3 expected = equivariant::Gal3::exp(increment * 0.02);
        const equivariant::Gal3 actual(zoh.delta_q.toRotationMatrix(), zoh.delta_v,
                                       zoh.delta_p, zoh.sum_dt);
        ok &= check(equivariant::Gal3::log(actual * expected.inverse()).norm() < 1e-12,
                    "equivariant interval uses left-endpoint ZOH measurement");
    }

    double pose_i[7];
    double speed_bias_i[9];
    double pose_j[7];
    double speed_bias_j[9];
    setPose(pose_i, Pi, Qi);
    setSpeedBias(speed_bias_i, Vi, ba, bg);
    setPose(pose_j, Pj, Qj);
    setSpeedBias(speed_bias_j, Vj, ba, bg);
    const double *parameters[4] = {pose_i, speed_bias_i, pose_j, speed_bias_j};
    double residuals[15];
    double jacobian_pose_i[15 * 7];
    double jacobian_speed_bias_i[15 * 9];
    double jacobian_pose_j[15 * 7];
    double jacobian_speed_bias_j[15 * 9];
    double *jacobians[4] = {jacobian_pose_i, jacobian_speed_bias_i,
                            jacobian_pose_j, jacobian_speed_bias_j};

    IMUFactor factor(&preintegration);
    ok &= check(factor.Evaluate(parameters, residuals, jacobians),
                mode ? "equivariant factor evaluation" : "classic factor evaluation");
    constexpr int benchmark_iterations = 200;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
        ok &= factor.Evaluate(parameters, residuals, jacobians);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() /
        benchmark_iterations;
    const Eigen::Map<const Eigen::Matrix<double, 15, 1>> residual(residuals);
    const Eigen::Map<const Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J0(jacobian_pose_i);
    const Eigen::Map<const Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1(jacobian_speed_bias_i);
    const Eigen::Map<const Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J2(jacobian_pose_j);
    const Eigen::Map<const Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J3(jacobian_speed_bias_j);
    ok &= check(residual.allFinite() && residual.norm() < 1e-5,
                "whitened factor residual finite and zero");
    ok &= check(J0.allFinite() && J1.allFinite() && J2.allFinite() && J3.allFinite(),
                "factor Jacobians finite");
    if (mode)
    {
        pose_j[0] += 0.15;
        pose_j[1] -= 0.08;
        speed_bias_j[0] += 0.12;
        speed_bias_j[4] += 0.004;
        speed_bias_j[8] -= 0.0008;
        ok &= check(factor.Evaluate(parameters, residuals, jacobians),
                    "nonzero equivariant factor evaluation");
        const FactorState nonzero_state = {
            std::vector<double>(pose_i, pose_i + 7),
            std::vector<double>(speed_bias_i, speed_bias_i + 9),
            std::vector<double>(pose_j, pose_j + 7),
            std::vector<double>(speed_bias_j, speed_bias_j + 9)};
        ok &= check(Eigen::Map<const Eigen::Matrix<double, 15, 1>>(residuals).norm() > 1e-3,
                    "nonzero state produces a nonzero residual");
        for (int column = 0; column < 6; ++column)
            ok &= checkJacobianColumn(factor, nonzero_state, 0, column, J0.col(column));
        for (int column = 0; column < 9; ++column)
            ok &= checkJacobianColumn(factor, nonzero_state, 1, column, J1.col(column));
        for (int column = 0; column < 6; ++column)
            ok &= checkJacobianColumn(factor, nonzero_state, 2, column, J2.col(column));
        for (int column = 0; column < 9; ++column)
            ok &= checkJacobianColumn(factor, nonzero_state, 3, column, J3.col(column));
    }
    std::cout << (mode ? "Equivariant" : "Classic")
              << " IMU factor evaluation: " << elapsed << " ms" << std::endl;
    return ok;
}
} // namespace

int main()
{
    const bool ok = testMode(0) && testMode(1);
    return ok ? 0 : 1;
}
