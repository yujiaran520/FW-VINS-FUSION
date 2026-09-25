# FW-VINS-FUSION V0.1

**V0.1 是 [V0](https://github.com/yujiaran520/FW-VINS-FUSION/tree/V0) 与
Gal(3) 等变预积分 V1 系列之间的工程过渡基线。**本分支保留 VINS-Fusion 原始
中点 IMU 预积分和 CPU Ceres 后端，不包含 V1.0/V1.1/V1.2 的等变算法。
默认分支 [`main`](https://github.com/yujiaran520/FW-VINS-FUSION) 继续开发
V1.2；需要复现本分支时请明确切换到 `V0.1`。

本项目面向扑翼飞行机器人的视觉惯性定位研究，基于
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 和
[ROS 2 Humble 移植版](https://github.com/fanhong-li/VINS-Fusion-ROS2-Humble)。
**仍是实验代码，不应直接用于安全关键飞行。**

## 版本定位

| 分支 | 预积分 | 重点 |
| --- | --- | --- |
| `V0` | 原始经典中点 | 上游 ROS 2 基线 |
| **`V0.1`** | **经典中点** | **FWAF 标定语义、CUDA 前端、诊断及可复现评估** |
| `V1.1` / `V1.2` / `main` | 可选 Gal(3) 等变 | 等变模型及后续稳定性研究；`V1.0` 为版本阶段而非已有远程分支 |

V0.1 不宣称比 V0 或 V1 更准确。FWAF-VID 全量试跑表明某些组合虽输出轨迹，
但仍发生严重尺度错误；应同时检查初始化、覆盖率、参考质量和尺度指标。

## 本版改进

- 六份 `config/FWAF-VID/` 配置直接包含 D435i/CUAV 前视、下倾 45 度
  单目/双目的 **Kalibr `T_cam_imu` 原始方向数值**、相机到 IMU 的时间偏移
  `td` 以及 imu_util/Allan 四组三轴噪声，均有中文来源和单位注释。VINS
  在内存中求完整刚体逆变换；调参直接改 YAML，不依赖原始标定文件。
- 显式支持连续时间 IMU 噪声密度，根据实测 `dt` 离散化；旧配置仍可按原有
  离散噪声语义加载。修正双目在线时间偏移 Jacobian 与可配置前端/后端频率。
- OpenCV CUDA 加速角点与光流、CPU Ceres 优化；提供时间戳、图像尺寸、
  重启/退出保护。GPU 模式需要具备 `cudaoptflow`、`cudaimgproc` 等模块的
  OpenCV 构建；当前代码不承诺普通系统 OpenCV 的纯 CPU 编译兼容。
- `diagnostics: 1` 可在输出目录写 `intermediate.csv`，记录逐后端帧 IMU
  区间、预积分 `dp/dv/dq`、协方差对角线、偏置及特征数。
- `scripts/` 可按序列和传感器独立运行 1.0x、无 RViz、OpenCV GPU/CPU Ceres
  的实验，并从轨迹与独立参考生成中文报告、同序列叠图和效能图。

标定方向、相机 rectified 内参及具体值见
[FWAF 配置说明](config/FWAF-VID/README.md)。相机图像为已矫正的
`image_rect_raw`，**不能**将原始 Kalibr radtan 内参直接套用。

## 构建与运行

需要 Ubuntu 22.04、ROS 2 Humble、Eigen3、Ceres 2.1、yaml-cpp、CUDA
OpenCV（含上述模块）及匹配该 OpenCV 的 `cv_bridge`。安装包或编译环境依
本机配置为准；单一 OpenCV/Ceres 运行时应避免 ABI 混用。

```bash
mkdir -p ~/fw_vins_ws/src
git clone --branch V0.1 https://github.com/yujiaran520/FW-VINS-FUSION.git \
  ~/fw_vins_ws/src/FW-VINS-FUSION
source /opt/ros/humble/setup.bash
cd ~/fw_vins_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/FWAF-VID/d435i_imu_stereo.yaml
```

配置中的 `output_path` 默认 `/tmp`，批处理会复制有效配置并设置独立输出
目录。`loop_fusion` 所需大型 DBoW 词袋不随新增工作提供；如使用回环，需
自行获取上游 `loop_fusion/support_files/brief_k10L6.bin`。

## FWAF-VID 实验与报告

**本仓库不包含 bag、GT 或既有实验结果。**将 FWAF ROS 2 bag 放到
`data/ROS2/FWAF-VID/<序列名>/`，并将动态序列的 TUM 参考放到
`data/ROS2/FWAF-VID/GT/full/`，作者短区间放到 `GT/eval/`；也可将
`FWAF_DATA_ROOT` 指向外部 FWAF 数据根目录。随仓库提供的
[`manifest.csv`](config/FWAF-VID/manifest.csv) 为该批已转换 bag 的话题
清单，其他转换版本需核对话题/持续时间。实际运行时必须有对应话题。

```bash
cd ~/fw_vins_ws/src/FW-VINS-FUSION
bash scripts/run_fwaf_v0_all.bash --dry-run
bash scripts/run_fwaf_v0_all.bash --resume
python3 scripts/report_fwaf_v0.py report
python3 scripts/evaluate_fwaf_v0.py
python3 scripts/test_evaluate_fwaf_v0.py
```

脚本需要 Python 3 的 NumPy、Matplotlib、PyYAML。运行结果默认放在
`results/v0_fwaf_full/`，含 `report.md`、`evaluation.md`、`accuracy.csv`、
`accuracy_eval.csv`、逐序列四条件对比图及逐运行 `vio.csv/intermediate.csv`。
自定义 ROS/依赖环境可通过 `FW_VINS_SETUP` 指向 setup 脚本，或用
`VINS_NODE_BINARY` 指定安装的本分支估计器二进制，避免使用其他版本节点。

既有本地全量实验用**数值等价的外部标定读取配置**运行 22 个序列、72 个有
话题组合：71 个有轨迹、1 个 `Indoor_16` 下倾单目未初始化，16 个因缺 IMU
话题无法运行；13 个有轨迹组合的输出跨度不足 bag 时长 80%。它们不是新内联
配置在 GitHub 上自动重新跑出的实验；新内联配置另经实包单目/双目烟测。

主评估使用 `GT/full` 加
[`evaluation_intervals.yaml`](config/FWAF-VID/evaluation_intervals.yaml)
记录的人工着陆去尾时间；作者 `GT/eval` 是另一个更短的选段，单独计算，
不可和主指标合并。ATE 使用固定尺度 SE(3) 位置配准，平移 RPE 为 1/5 秒
间隔，Sim(3) 比例只用于**尺度诊断**。室外位置来自非 RTK-fixed GNSS，
室内 UWB 存在严重跳变，静态扑翼没有本地位置真值；这些数值是参考一致性
而非动捕级精度，室内/静态不能据此宣称绝对定位性能。

## 来源与许可

本项目保留上游 VINS-Fusion 和 ROS 2 移植版作者的代码版权与许可声明。
使用 Ceres、DBoW2、camodocal 相机模型及 GeographicLib 等第三方组件时
应分别遵循其许可。本项目按 [GPL-3.0](LICENCE) 发布，软件按原样提供，
不附带适航、安全性或定位精度保证。
