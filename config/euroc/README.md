# EuRoC Configuration

`euroc_mono_imu_config.yaml` and `euroc_stereo_imu_config.yaml` contain the
parameters used by the current estimator. Review topic names and output paths
before running them against a ROS 2 EuRoC bag.

The classic `acc_n`, `gyr_n`, `acc_w`, and `gyr_w` values are used by the
original VINS midpoint covariance. The separately named `equivariant_*`
parameters are interpreted as continuous-time densities by the Gal(3)
implementation. Select the implementation with:

```yaml
equivariant_preintegration_enable: 0  # classic midpoint
equivariant_preintegration_enable: 1  # Gal(3) equivariant
```

Run the monocular-IMU example after building and sourcing the workspace:

```bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/euroc/euroc_mono_imu_config.yaml
```
