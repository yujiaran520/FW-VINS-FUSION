#!/usr/bin/env python3
"""Sequential FWAF-VID v0 run ledger and evidence-only report."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config/FWAF-VID/manifest.csv"
DEFAULT_OUTPUT = ROOT / "results/v0_fwaf_full"
BENCHMARK = ROOT / "scripts/benchmark_v0_gpu.bash"
AUTHOR_README = "FWAF-VID dataset README, Dataset Sequences list"
# Dataset-author camera perspectives; lowercase 03/04 correspond to static 21/22.
MOUNT = {
    "Indoor_13": "45deg", "Indoor_14": "45deg", "Indoor_15": "forward",
    "Indoor_16": "45deg", "Indoor_17": "forward",
    "Outdoor_01": "45deg", "Outdoor_02": "45deg", "Outdoor_03": "forward",
    "Outdoor_04": "45deg", "Outdoor_05": "45deg", "Outdoor_06": "forward",
    "Outdoor_07": "45deg", "Outdoor_08": "forward", "Outdoor_09": "45deg",
    "Outdoor_10": "45deg", "Outdoor_11": "forward", "Outdoor_12": "forward",
    "Static_flapping_18": "forward", "Static_flapping_19": "forward",
    "Static_flapping_20": "forward", "static_flapping_03": "45deg",
    "static_flapping_04": "45deg",
}


def manifest_rows():
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["sequence"] for row in rows]
    if len(rows) != 22 or len(set(names)) != 22 or set(names) != set(MOUNT):
        raise ValueError("Manifest sequences differ from the explicit author mounting map")
    return rows


def jobs(rows):
    for row in rows:
        for sensor, profile, field in (
            ("camera", "d435i", "camera_imu"),
            ("mavros", "cuav-" + MOUNT[row["sequence"]], "mavros_imu"),
        ):
            for mode in ("mono", "stereo"):
                available = int(row[field]) > 0 and int(row["left_images"]) > 0
                if mode == "stereo":
                    available = available and int(row["right_images"]) > 0
                yield {**row, "sensor": sensor, "profile": profile, "mode": mode,
                       "key": f"{row['sequence']}__{profile}__{mode}",
                       "available": available,
                       "reason": "" if available else f"missing {field} or required image topic in manifest"}


def state_path(output, job):
    return output / "state" / (job["key"] + ".json")


def load_state(output, job):
    path = state_path(output, job)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(output, job, state):
    path = state_path(output, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def valid_success(state):
    if state.get("status") != "SUCCESS":
        return False
    run = Path(state.get("run_dir", "/nonexistent"))
    return ((run / "status.txt").exists()
            and (run / "status.txt").read_text().strip() == "SUCCESS"
            and (run / "summary.csv").is_file() and (run / "vio.csv").is_file()
            and (run / "vio.csv").stat().st_size > 0)


def run_batch(args, all_jobs):
    available = [job for job in all_jobs if job["available"]]
    if args.dry_run:
        count = 0
        for job in all_jobs:
            if not job["available"]:
                print(f"SKIP {job['key']}: {job['reason']}")
            elif valid_success(load_state(args.output, job)):
                print(f"SUCCESS (resume skip) {job['key']}")
            elif args.limit is None or count < args.limit:
                print(f"PLAN {job['key']} opencv rate=1.0 rviz=0")
                count += 1
        print(f"manifest={len(MOUNT)} eligible={len(available)} missing={len(all_jobs)-len(available)} planned={count}")
        return 0
    if not args.resume and any(state_path(args.output, job).exists() for job in available):
        raise ValueError("Existing run state found; use --resume to skip successes and retry failures")
    args.output.mkdir(parents=True, exist_ok=True)
    launched = 0
    failures = 0
    for job in available:
        old = load_state(args.output, job)
        if valid_success(old):
            print(f"SUCCESS (resume skip) {job['key']}", flush=True)
            continue
        if args.limit is not None and launched >= args.limit:
            break
        run = (args.output / "runs" / job["sequence"] / job["profile"] / job["mode"]
               / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]))
        state = {"status": "RUNNING", "run_dir": str(run), "attempts": old.get("attempts", [])
                 + ([{k: old.get(k) for k in ("status", "run_dir", "exit_code")}]
                    if old else []), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        save_state(args.output, job, state)
        env = os.environ.copy()
        env["VINS_BENCHMARK_RUN_DIR"] = str(run)
        timeout = max(180, math.ceil(float(job["duration_s"]) * 3 + 120))
        print(f"RUN {job['key']} -> {run} (timeout {timeout}s)", flush=True)
        launched += 1
        status = "FAILED"
        code = None
        proc = None
        try:
            proc = subprocess.Popen(["bash", str(BENCHMARK), job["mode"], "opencv",
                                     job["sequence"], job["profile"]], env=env,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = proc.returncode
            if status != "TIMEOUT":
                status = "SUCCESS" if code == 0 and valid_success(
                    {"status": "SUCCESS", "run_dir": str(run)}) else "FAILED"
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            code = proc.returncode if proc else None
            raise
        finally:
            state.update(status=status, exit_code=code,
                         finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            save_state(args.output, job, state)
            print(f"{status} {job['key']} {run}", flush=True)
        if status != "SUCCESS":
            failures += 1
    report(args.output, all_jobs)
    print(f"launched={launched} failures={failures} report={args.output / 'report.md'}")
    return 1 if failures else 0


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (ValueError, TypeError):
        return math.nan


def fmt(value, digits=2):
    return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"


def measure(job, run):
    import numpy as np

    result = {"rows": 0, "span_s": math.nan, "coverage": math.nan,
              "monotonic": "N/A", "initialization": "NO", "restarts": "N/A",
              "cpu_percent": math.nan, "rss_mb": math.nan, "gpu_percent": math.nan,
              "solver_ms": math.nan, "flags": [], "trajectory": None}
    summary = run / "summary.csv"
    if summary.is_file():
        with summary.open(newline="") as stream:
            metrics = {r["metric"]: r["value"] for r in csv.DictReader(stream)}
        for key, metric, scale in (("cpu_percent", "estimator_cpu_average", 1),
                                   ("rss_mb", "estimator_rss_maximum", 1024),
                                   ("gpu_percent", "gpu_utilization_average", 1),
                                   ("solver_ms", "solver_average", 1)):
            result[key] = number(metrics.get(metric)) / scale
        if metrics.get("estimator_alive_after_playback") != "1":
            result["flags"].append("估计器提前退出")
    log = run / "vins.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        result["restarts"] = str(len(re.findall(
            r"failure detection|reboot|restart|reset estimator", text, re.I)))
        if int(result["restarts"]) > 0:
            result["flags"].append("日志疑似重启/故障")
        if "Not enough features or parallax" in text and "Initialization finish!" not in text:
            result["flags"].append("特征或视差不足，未能初始化")
    trajectory = run / "vio.csv"
    if not trajectory.is_file() or trajectory.stat().st_size == 0:
        result["flags"].append("无轨迹/未初始化")
        return result
    try:
        with trajectory.open(newline="") as stream:
            data = np.array([[float(x) for x in row[:4]] for row in csv.reader(stream)
                             if len(row) >= 4], dtype=float).reshape(-1, 4)
        if not len(data) or not np.isfinite(data).all():
            raise ValueError("empty or non-finite trajectory")
    except (ValueError, OSError) as exc:
        result["flags"].append(f"轨迹无效: {exc}")
        return result
    result["trajectory"] = data
    result["rows"] = len(data)
    result["initialization"] = "YES"
    result["monotonic"] = "YES" if np.all(np.diff(data[:, 0]) > 0) else "NO"
    result["span_s"] = float(data[-1, 0] - data[0, 0])
    result["coverage"] = result["span_s"] / float(job["duration_s"])
    if result["monotonic"] != "YES":
        result["flags"].append("时间戳非严格递增")
    if result["coverage"] < 0.8 or result["coverage"] > 1.1:
        result["flags"].append("轨迹时间跨度偏离 bag 时长")
    if result["rows"] < 10:
        result["flags"].append("轨迹点过少")
    return result


def report(output, all_jobs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    records = []
    for job in all_jobs:
        state = load_state(output, job) if job["available"] else {}
        status = ("SKIP" if not job["available"] else
                  "SUCCESS" if valid_success(state) else state.get("status", "PENDING"))
        if status == "SUCCESS" and state.get("status") != "SUCCESS":
            status = "FAILED"
        run = Path(state["run_dir"]) if state.get("run_dir") else None
        metrics = measure(job, run) if run and run.is_dir() else None
        flags = metrics["flags"] if metrics else []
        if job["available"] and state.get("status") == "SUCCESS" and status != "SUCCESS":
            flags.append("成功标记或结果文件缺失")
        record = {"sequence": job["sequence"], "sensor": job["sensor"],
                  "mount": MOUNT[job["sequence"]], "profile": job["profile"],
                  "mode": job["mode"], "compute": "opencv", "rate": "1.0",
                  "reference": job["reference"], "status": status,
                  "reason": job["reason"], "run_dir": str(run) if run else "",
                  "rows": metrics["rows"] if metrics else "",
                  "span_s": fmt(metrics["span_s"]) if metrics else "",
                  "coverage": fmt(metrics["coverage"], 3) if metrics else "",
                  "monotonic": metrics["monotonic"] if metrics else "",
                  "initialization": metrics["initialization"] if metrics else "",
                  "restarts": metrics["restarts"] if metrics else "",
                  "cpu_percent": fmt(metrics["cpu_percent"]) if metrics else "",
                  "rss_mb": fmt(metrics["rss_mb"]) if metrics else "",
                  "gpu_percent": fmt(metrics["gpu_percent"]) if metrics else "",
                  "solver_ms": fmt(metrics["solver_ms"]) if metrics else "",
                  "flags": "; ".join(flags)}
        records.append(record)
        if metrics and metrics["trajectory"] is not None:
            data = metrics["trajectory"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(data[:, 1], data[:, 2], linewidth=1)
            axes[0].set(xlabel="VIO x (m)", ylabel="VIO y (m)", title="Estimated XY")
            axes[0].axis("equal")
            for index, label in enumerate(("x", "y", "z"), start=1):
                axes[1].plot(data[:, 0] - data[0, 0], data[:, index], label=label)
            axes[1].set(xlabel="VIO elapsed time (s)", ylabel="Position (m)", title="Estimated position")
            axes[1].legend()
            fig.suptitle(job["key"] + " (not aligned to reference)", fontsize=10)
            fig.tight_layout()
            fig.savefig(figures / (job["key"] + "_trajectory.png"), dpi=120)
            plt.close(fig)
    with (output / "manifest_status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    completed = [r for r in records if r["status"] == "SUCCESS"]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(completed) * 0.32), 7))
    if completed:
        labels = [r["sequence"] + " " + r["sensor"][0] + "/" + r["mode"][0]
                  for r in completed]
        x = np.arange(len(labels))
        for ax, field, title in ((axes[0], "cpu_percent", "Estimator CPU (%)"),
                                 (axes[1], "rss_mb", "Peak estimator RSS (MiB)")):
            ax.bar(x, [number(r[field]) for r in completed])
            ax.set_ylabel(title)
            ax.set_xticks(x, labels, rotation=90, fontsize=7)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No successful runs", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(figures / "performance.png", dpi=120)
    plt.close(fig)
    groups = [(sensor, mode) for sensor in ("camera", "mavros") for mode in ("mono", "stereo")]
    fig, ax = plt.subplots(figsize=(9, 4))
    group_values = [[number(r["coverage"]) for r in completed
                     if r["sensor"] == sensor and r["mode"] == mode]
                    for sensor, mode in groups]
    for index, values in enumerate(group_values):
        if values:
            ax.scatter([index] * len(values), values, alpha=0.45, s=18)
            ax.plot(index, np.median(values), marker="_", color="black", markersize=22)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="0.8 coverage warning")
    ax.set_xticks(range(len(groups)), ["/".join(group) for group in groups])
    ax.set(ylabel="Trajectory span / bag duration", title="Coverage by IMU and camera mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "coverage.png", dpi=120)
    plt.close(fig)
    example = next((r for r in records if r["sequence"] == "Outdoor_08"
                    and r["profile"] == "d435i" and r["mode"] == "stereo"
                    and r["status"] == "SUCCESS"), None)
    diagnostic_figure = False
    if example:
        trace = Path(example["run_dir"]) / "intermediate.csv"
        if trace.is_file():
            with trace.open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            if samples:
                t0 = number(samples[0]["image_time"])
                seconds = [number(row["image_time"]) - t0 for row in samples]
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                for field in ("dt_sum", "preint_sum_dt"):
                    axes[0].plot(seconds, [number(row[field]) for row in samples], label=field)
                axes[0].set_ylabel("IMU interval (s)")
                axes[0].legend()
                axes[1].plot(seconds, [number(row["features"]) for row in samples])
                axes[1].set_ylabel("Tracked feature IDs")
                for field in ("cov_p_x", "cov_rot_x", "cov_v_x"):
                    axes[2].plot(seconds, [number(row[field]) for row in samples], label=field)
                axes[2].set(xlabel="Bag elapsed time (s)", ylabel="Preintegration covariance")
                axes[2].set_yscale("symlog", linthresh=1e-12)
                axes[2].legend()
                fig.suptitle("Outdoor_08 D435i stereo, frame-boundary diagnostics")
                fig.tight_layout()
                fig.savefig(figures / "preintegration_example.png", dpi=120)
                plt.close(fig)
                diagnostic_figure = True
    counts = {s: sum(r["status"] == s for r in records)
              for s in ("SUCCESS", "FAILED", "TIMEOUT", "INTERRUPTED", "RUNNING", "PENDING", "SKIP")}
    short = [r for r in completed if number(r["coverage"]) < 0.8]
    lines = ["# FWAF-VID v0 OpenCV 顺序实验", "",
             f"来源：`{MANIFEST.relative_to(ROOT)}`；安装角度：`{AUTHOR_README}`。",
             "仅统计本目录 ledger 中的新运行；每个组合单目/双目、1.0 倍速、无 RViz。", "",
             "## 状态", "",
             " | ".join(f"{key}: {value}" for key, value in counts.items()), "",
              "SKIP 表示 manifest 中相应 IMU 或图像话题数量为 0，并非运行结果。PENDING 表示尚未执行。", "",
              "## 结果要点", "",
              f"可运行组合共 {len(all_jobs) - counts['SKIP']} 个；其中 {counts['SUCCESS']} 个有轨迹且干净退出，"
              f"{counts['FAILED']} 个失败。成功组合中 {len(short)} 个轨迹跨度不足 bag 时长的 80%，"
              "不能仅凭退出码认为整段稳定追踪。", "",
              "| IMU | 相机 | 有轨迹组合 | 覆盖率中位数 | 低于 80% |",
              "|---|---|---:|---:|---:|",
              *[f"| {sensor} | {mode} | {len(values)} | {fmt(float(np.median(values)), 3) if values else 'N/A'} | "
                f"{sum(value < 0.8 for value in values)} |"
                for (sensor, mode), values in zip(groups, group_values)], "",
              "## 解读限制", "",
             "轨迹图仅为未对齐的 VIO 估计位置，不能视为真实轨迹或精度证明。",
             "coverage 为 VIO 首末时间戳跨度 / bag 时长，不测量起止对齐或轨迹连续性；初始化仅表示有有效轨迹点。",
             "日志重启计数基于关键词启发式，缺失日志显示 N/A；时间戳、跨度、日志异常见标记列。",
             "室外 GNSS ENU + MAVROS 姿态不是 RTK fixed 真值；Indoor UWB pos_3d 存在大跳变且无可靠姿态真值；",
              "静态扑翼序列没有 GNSS/UWB 位姿话题。这里不计算 ATE/RPE，也不宣称绝对定位精度。",
              "CUAV 输入是 MAVROS 发布的滤波 IMU，而 Allan 标定来自传感器采样；噪声一致性需另行核验。", "",
             "## 运行记录", "",
             "| 序列 | IMU / 安装 | 模式 | 状态 | 轨迹点 | 跨度(s) | coverage | 单调 | 初始化 | 重启 | CPU % | RSS MiB | 标记 / 原因 |",
             "|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|"]
    for r in records:
        fields = [r["sequence"], r["sensor"] + "/" + r["mount"], r["mode"],
                  r["status"], str(r["rows"]), r["span_s"], r["coverage"],
                  r["monotonic"], r["initialization"], r["restarts"],
                  r["cpu_percent"], r["rss_mb"], r["flags"] or r["reason"]]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in fields) + " |")
    lines += ["", "## 图表", "", "![轨迹时间覆盖率](figures/coverage.png)", "",
              "![资源性能](figures/performance.png)", ""]
    if diagnostic_figure:
        lines += ["Outdoor_08 D435i 双目的逐帧中间量样例（原始 CSV 可查每次运行）：", "",
                  "![预积分诊断](figures/preintegration_example.png)", ""]
    lines += ["同序列参考轨迹对齐、ATE/RPE/尺度和效率比较见 [位置参考评估](evaluation.md)。", "",
              "各个有轨迹的运行图：`figures/*_trajectory.png`。原始运行目录见 `manifest_status.csv`。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report={output / 'report.md'} success={counts['SUCCESS']} pending={counts['PENDING']} skip={counts['SKIP']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    batch = sub.add_parser("batch", help="run eligible jobs sequentially")
    batch.add_argument("--dry-run", action="store_true", help="print plan without writing or launching")
    batch.add_argument("--limit", type=int, help="maximum new runs (not bags or skips)")
    batch.add_argument("--resume", action="store_true", help="skip valid successes; retry failures")
    batch.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    rpt = sub.add_parser("report", help="regenerate report from this batch ledger")
    rpt.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "batch" and args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    try:
        all_jobs = list(jobs(manifest_rows()))
        if args.command == "batch":
            return run_batch(args, all_jobs)
        report(args.output, all_jobs)
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
