# FW-VINS-FUSION

FW-VINS-FUSION is a ROS 2 Humble visual-inertial state estimator derived from
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). It targets
ARM64/NVIDIA Jetson platforms and adds an optional Gal(3) equivariant IMU
preintegration model alongside the original midpoint preintegration.

## Features

- Monocular camera + IMU, stereo camera + IMU, and stereo-only estimation.
- Runtime-selectable classic or Gal(3) equivariant IMU preintegration.
- CPU feature tracking or CUDA-accelerated OpenCV feature tracking.
- Online camera-IMU extrinsic and temporal calibration.
- Visual loop closure and optional GPS global fusion.
- EuRoC and FWAF-VID configuration examples.

The ROS package names remain `vins`, `camera_models`, `loop_fusion`, and
`global_fusion` for compatibility with existing launch files and tools.

## Requirements

- Ubuntu 22.04 and ROS 2 Humble
- CMake 3.5 or newer and a C++14 compiler
- Eigen3
- Ceres Solver 2.1 or newer
- OpenCV and ROS 2 `cv_bridge`
- Optional: CUDA-enabled OpenCV for GPU feature tracking

## Build

Clone the repository into a ROS 2 workspace:

```bash
mkdir -p ~/fw_vins_ws/src
cd ~/fw_vins_ws/src
git clone https://github.com/yujiaran520/FW-VINS-FUSION.git
cd ..
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DVINS_ENABLE_GPU=OFF
source install/setup.bash
```

Set `VINS_ENABLE_GPU=ON` only when OpenCV was built with the CUDA modules
required by `vins/CMakeLists.txt`.

## Run

Start the estimator with a configuration file:

```bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/euroc/euroc_mono_imu_config.yaml
```

Start RViz separately when visualization is needed:

```bash
ros2 launch vins vins_rviz.launch.xml
```

Datasets are intentionally not included. Play a compatible ROS 2 bag in a
separate terminal and adjust topic names and calibration values in the selected
configuration file.

## IMU Preintegration

Select the implementation in an EuRoC or FWAF-VID configuration:

```yaml
# Original VINS midpoint preintegration
equivariant_preintegration_enable: 0

# Gal(3) equivariant preintegration
equivariant_preintegration_enable: 1
```

The equivariant implementation uses separate continuous-time IMU noise density
and bias random-walk parameters. See `config/FWAF-VID/README.md` and the example
YAML files for details.

## Loop Vocabulary

This source-only repository omits the approximately 58 MB DBoW vocabulary. To
use `loop_fusion`, download the upstream vocabulary after cloning:

```bash
curl -L \
  https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/loop_fusion/support_files/brief_k10L6.bin \
  -o loop_fusion/support_files/brief_k10L6.bin
```

The file is ignored by Git and will not be uploaded in later commits.

## Repository Scope

The repository contains algorithm source code, ROS package metadata, tests,
launch files, calibration/configuration examples, and the license. Datasets,
build/install outputs, logs, experiment results, downloaded dependencies,
calibration image sets, visualization models, and large binary assets are
excluded.

## Synchronizing Updates

Before starting work, synchronize with GitHub:

```bash
git switch main
git pull --rebase origin main
```

After changing source code, review and publish only intended files:

```bash
git status
git diff
git add vins camera_models loop_fusion global_fusion config README.md .gitignore
git commit -m "Describe the change"
git push origin main
```

Do not use `git add -f` for ignored data or generated files. If GitHub rejects a
push because the remote changed, run `git pull --rebase origin main`, resolve
any conflicts, and push again.

## Acknowledgements

This project is based on the original
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) and the
[ROS 2 Humble ARM port](https://github.com/JanekDev/VINS-Fusion-ROS2-humble-arm).
It also incorporates camera-model, DBoW2, and GeographicLib components retained
from those projects. Please cite the original VINS publications when using this
software in academic work.

## License

Released under GPL-3.0. See `LICENCE`.
