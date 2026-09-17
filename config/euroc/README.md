# EuRoC Configuration

`euroc_mono_imu_config.yaml` and `euroc_stereo_imu_config.yaml` contain the
parameters used by the current estimator. Review topic names and output paths
before running them against a ROS 2 EuRoC bag.

In the in-progress v1.2 implementation, classic midpoint and Gal(3)
preintegration both read `acc_n`, `gyr_n`, `acc_w`, and `gyr_w` as
continuous-time densities. Each configuration must declare the semantics and
the bias repropagation thresholds:

```yaml
imu_noise_semantics: "continuous_time_density"
bias_acc_repropagation_threshold: 0.1
bias_gyr_repropagation_threshold: 0.01
```

The former separately named `equivariant_*` noise parameters are no longer
used. Select the implementation with:

```yaml
equivariant_preintegration_enable: 0  # classic midpoint
equivariant_preintegration_enable: 1  # Gal(3) equivariant
```

Run the monocular-IMU example after building and sourcing the workspace:

```bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/euroc/euroc_mono_imu_config.yaml
```
