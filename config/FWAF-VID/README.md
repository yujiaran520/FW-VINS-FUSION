# FWAF-VID V0.1 可调配置

这些文件用于在原始中点预积分 VINS-Fusion ROS 2 上运行同步后的 FWAF-VID
数据。六份估计器配置均启用 OpenCV CUDA 特征跟踪，并保留 CPU Ceres 后端。
`Outdoor_08` 四组合基准显示，Ceres CUDA 在小规模滑窗问题上约慢 2 至 3 倍；
OpenCV CUDA 则在完整实时处理的同时，将单目/双目 CPU 占用分别从约
`112%/149%` 降至 `65%/61%`，因此最终使用 `use_gpu=1`、
`use_gpu_acc_flow=1`、`use_gpu_ceres=0`。

## 选择配置

| 相机安装方式 | IMU | 单目 | 双目 |
| --- | --- | --- | --- |
| 任意 D435i 安装方式 | D435i 内置 | `d435i_imu_mono.yaml` | `d435i_imu_stereo.yaml` |
| 前视 | CUAV V5+ | `cuav_imu_forward_mono.yaml` | `cuav_imu_forward_stereo.yaml` |
| 下倾 45 度 | CUAV V5+ | `cuav_imu_45deg_mono.yaml` | `cuav_imu_45deg_stereo.yaml` |

数据集 README 标明了各序列的安装方式。D435i 内置 IMU 与相机刚性连接，
因此不需要单独区分前视和下倾配置。

## 标定映射

- ROS bag 输入为 RealSense 已矫正的 `image_rect_raw`。`left.yaml` 和
  `right.yaml` 使用数据集原始 VINS 配置提供的共同 rectified 内参和零畸变；
  不再对图像重复应用 Kalibr 的 radtan 畸变模型。
- **六份运行配置直接写参数数值**，不需要依赖标定文件路径。`T_cam_imu0/1`
  是从对应 Kalibr camchain 原样摘出的 4×4 数值，方向 IMU→相机，平移单位 m；
  VINS 内部只负责取完整逆矩阵成为相机→IMU/body 的外参。调外参直接修改矩阵，
  不要把其平移当作 `body_T_cam` 的平移。旧配置仍可用 `body_T_cam*`。
- `acc_n/gyr_n/acc_w/gyr_w` 在 YAML 内直接写 imu_util/Allan `x/y/z` 三轴
  原始值；`imu_noise_is_density: 1` 声明为连续时间密度，预积分按实际 `dt`
  离散化。CUAV 输入由 MAVROS 发布，滤波后噪声未必等于原始采样标定。
- `td` 在 YAML 内直接写 Kalibr cam0 `timeshift_cam_imu`（秒），
  符号是 `t_imu = t_image + td`。VINS 仅支持一个 `td`；双目 cam1 的不同值
  在配置注释里列明，不能错误地当成已独立补偿。
- 如需恢复之前的外部文件直读方式，仍可选择互斥的 `kalibr_camchain` 和
  `imu_allan` / `kalibr_imu`；这只是可选入口，不是 FWAF 运行配置的依赖。
- `focal_length` 使用 rectified 焦距，使关键帧视差、RANSAC 阈值和视觉残差
  权重与 FWAF-VID 的像素尺度一致。
- 输入话题使用数据集同步话题表，而不是运行 Kalibr 时使用的临时话题名称。
- 批处理会复制本文件为 `effective_config.yaml` 留证；本仓库不含本地旧实验
  的 bag/结果。先前 72 次实验的旧配置从外部标定文件读取相同数值，不能把
  它们标成已使用 V0.1 内联配置重新执行。
- 临时诊断开关 `diagnostics: 1` 在 `output_path/intermediate.csv` 输出每帧对齐
  时间、IMU 样本/间隔、预积分 `dp/dv/dq`、协方差对角项、偏置及特征数。
  该日志在后端帧级记录，非每次 Ceres 因子评估；批处理默认开启，可设置
  `VINS_DIAGNOSTICS=0` 关闭。诊断也会产生少量性能开销。

将 bag 放到仓库 `data/ROS2/FWAF-VID/`（或设置 `FWAF_DATA_ROOT`），从仓库
根目录运行全量顺序实验（仅实际存在话题的组合）：

```bash
bash scripts/run_fwaf_v0_all.bash --dry-run
bash scripts/run_fwaf_v0_all.bash --resume
python3 scripts/report_fwaf_v0.py report
python3 scripts/evaluate_fwaf_v0.py
```

该脚本按数据集作者安装视角选择 CUAV 前视/下倾 45 度配置，D435i 使用内置
IMU 标定；所有可用组合均单目/双目、1.0x、无 RViz、OpenCV GPU、CPU Ceres。
缺失话题记为 SKIP。主精度协议使用 `GT/full` 按
`config/FWAF-VID/evaluation_intervals.yaml` 的逐序列着陆去尾值截断；
`GT/eval` 作者短区间另列，不与主协议混算。报告的 ATE/RPE 都只对位置评估：
室外 GNSS 非 RTK fixed，室内 UWB 存在大跳变，静态扑翼缺位姿真值。

运行示例：

```bash
source ~/fw_vins_ws/install/setup.bash
ros2 run vins vins_node \
  ~/fw_vins_ws/src/FW-VINS-FUSION/config/FWAF-VID/d435i_imu_stereo.yaml
```
