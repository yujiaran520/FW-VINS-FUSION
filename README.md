# FW-VINS-FUSION

FW-VINS-FUSION（Flapping-Wing VINS-Fusion）是面向**扑翼飞行机器人飞行场景**的
视觉惯性导航算法改进项目，基于
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 及其 ROS 2
移植版本开发，因此命名为 **FW-VINS-FUSION**。

> **开发状态：本项目正在持续改进，目前仅完成 Gal(3) 等变 IMU 预积分这一项
> 算法改进。现有代码和实验结果仅供学习、研究与方案参考，不代表最终版本，
> 不建议直接用于生产环境或安全关键飞行任务。**

## 当前改进

目前实现了可选的 **Gal(3) 等变 IMU 预积分**，并保留原始 VINS-Fusion
中点预积分作为基线。用户可以通过配置文件在两种实现之间切换：

```yaml
# 原始 VINS-Fusion 中点预积分
equivariant_preintegration_enable: 0

# Gal(3) 等变 IMU 预积分
equivariant_preintegration_enable: 1
```

该实现参考 G. Delama 等人的论文 *Equivariant IMU Preintegration With
Biases: A Galilean Group Approach*（IEEE Robotics and Automation Letters，
2025），使用 Eigen 独立实现，不依赖额外李群库。

### 潜在优势

- 在 Gal(3) 李群上统一表示和更新姿态、速度、位置及时间增量，更好地保留状态的
  几何结构，避免将旋转简单视为欧氏空间中的加法量。
- 协方差和 IMU 偏置雅可比随群动力学共同传播，使状态增量、噪声传播和偏置修正
  使用一致的数学框架。
- 对坐标系变换具有等变结构，理论上有助于降低线性化结果对参考坐标选择的敏感性，
  改善估计一致性。
- 单独配置连续时间陀螺仪、加速度计噪声密度及偏置随机游走参数，避免切换预积分
  模型时误用原始 VINS 参数的离散噪声语义。
- 面向扑翼机器人可能出现的快速姿态变化、高动态运动和周期性机体振动，该几何建模
  方式具有进一步研究鲁棒性和一致性的潜力。

上述内容是基于模型结构的理论优势，并不表示该实现已经在所有数据集或真实扑翼平台
上获得更高精度。当前仍需开展更充分的消融实验、参数标定、公开数据集评估和实机飞行
验证。

## 基础能力

除上述 IMU 预积分改进外，其余功能主要继承自 VINS-Fusion 及 ROS 2 移植版本：

- 单目相机 + IMU、双目相机 + IMU 及纯双目估计。
- 在线相机-IMU 外参和时间偏移标定。
- 视觉回环与可选 GPS 全局融合。
- CPU 特征跟踪或基于 CUDA OpenCV 的特征跟踪。
- ROS 2 Humble 和 ARM64/NVIDIA Jetson 运行支持。
- EuRoC 与 FWAF-VID 配置示例。

为兼容现有启动文件和工具，ROS 包名仍保留为 `vins`、`camera_models`、
`loop_fusion` 和 `global_fusion`。

## 环境要求

- Ubuntu 22.04 与 ROS 2 Humble
- CMake 3.5 或更高版本及支持 C++14 的编译器
- Eigen3
- Ceres Solver 2.1 或更高版本
- OpenCV 与 ROS 2 `cv_bridge`
- 可选：支持所需 CUDA 模块的 OpenCV

## 编译

将仓库克隆到 ROS 2 工作空间：

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

只有在 OpenCV 已编译 `vins/CMakeLists.txt` 中要求的 CUDA 模块时，才应设置
`VINS_ENABLE_GPU=ON`。

## 运行

使用配置文件启动估计器：

```bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/euroc/euroc_mono_imu_config.yaml
```

需要可视化时可单独启动 RViz：

```bash
ros2 launch vins vins_rviz.launch.xml
```

本仓库不包含数据集。请在另一个终端播放兼容的 ROS 2 bag，并根据传感器修改所选
配置文件中的话题名称、标定参数和噪声参数。

## 回环词袋

为保持仓库仅包含关键代码，本仓库未上传约 58 MB 的 DBoW 词袋。需要使用
`loop_fusion` 时，可在克隆后下载上游词袋：

```bash
curl -L \
  https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/loop_fusion/support_files/brief_k10L6.bin \
  -o loop_fusion/support_files/brief_k10L6.bin
```

该文件已被 `.gitignore` 排除，后续提交不会上传它。

## 仓库范围

仓库仅包含算法源码、ROS 包元数据、测试、启动文件、配置示例和许可证。数据集、
编译/安装产物、日志、实验结果、下载的依赖、标定图片集、可视化模型及大型二进制
资源均不纳入版本控制。

## 同步更新

开始修改前先同步远程仓库：

```bash
git switch main
git pull --rebase origin main
```

修改后检查并上传必要文件：

```bash
git status
git diff
git add vins camera_models loop_fusion global_fusion config README.md .gitignore
git commit -m "描述本次修改"
git push origin main
```

不要使用 `git add -f` 强制加入被忽略的数据或生成文件。如果远程已有新提交，先执行
`git pull --rebase origin main`，解决可能的冲突后再推送。

## 上游项目与引用

本项目派生自：

- [HKUST Aerial Robotics Group/VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- [JanekDev/VINS-Fusion-ROS2-humble-arm](https://github.com/JanekDev/VINS-Fusion-ROS2-humble-arm)

仓库还保留了来自上游项目的相机模型、DBoW2 和 GeographicLib 相关代码。使用本项目
开展学术研究时，请同时引用原始 VINS-Fusion/VINS-Mono 论文以及实际使用的等变
预积分论文，并保留相应作者和版权声明。

## 许可证与免责声明

本项目按照 **GNU General Public License v3.0（GPL-3.0）** 发布，完整条款见
[`LICENCE`](LICENCE)。在 GPL-3.0 允许的范围内，你可以使用、研究、修改和再分发
代码；分发修改版本或二进制版本时，需要遵守 GPL-3.0 的源代码提供、许可证保留及
同许可证传播等要求。

项目中源自第三方组件的代码仍受其各自版权声明和许可证条款约束。使用者有责任核对
并遵守相关条款。本软件按“原样”提供，不附带任何明示或默示担保；作者及贡献者不对
因使用本软件造成的飞行事故、设备损坏、数据丢失或其他直接、间接损失承担责任。
本节仅用于说明项目许可，不构成法律意见。
