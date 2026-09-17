# FWAF-VID Configuration

This directory provides RealSense infrared camera and IMU calibration examples
for monocular-IMU and stereo-IMU FWAF-VID runs. Dataset bags are not included.
Update topic names, calibration values, and output paths for your local setup
before running the estimator.

## Compute Mode

The supplied configurations target real-time processing on an NVIDIA Jetson
Orin. A CUDA-enabled OpenCV build accelerates corner detection and optical flow,
while Ceres runs the sliding-window optimization on the CPU. Build with
`VINS_ENABLE_GPU=OFF` and disable the GPU options in YAML when CUDA OpenCV is
not available.

## IMU Preintegration

The default is the original VINS midpoint preintegration:

```yaml
equivariant_preintegration_enable: 0
```

Set the value to `1` to use Gal(3) equivariant preintegration. It uses a
left-endpoint zero-order hold for each IMU interval. Both implementations use
the same continuous-time density parameters:

```yaml
imu_noise_semantics: "continuous_time_density"
acc_n: 0.1
gyr_n: 0.01
acc_w: 0.001
gyr_w: 0.0001
```

The densities are discretized using each measured IMU interval. The former
`equivariant_*` noise keys are not used by v1.2.

## Run

After building and sourcing the workspace:

```bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/FWAF-VID/realsense_mono_imu_config.yaml
```

Use `realsense_stereo_imu_config.yaml` for stereo-IMU estimation.
