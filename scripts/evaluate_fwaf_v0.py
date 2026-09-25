#!/usr/bin/env python3
"""Read-only evaluation of the existing FWAF-VID v0 run ledger."""

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FWAF_DATA_ROOT", str(ROOT / "data/ROS2/FWAF-VID")))
CONFIG = ROOT / "config/FWAF-VID/evaluation_intervals.yaml"
VIO_GAP = 0.25
GT_GAP = {"indoor": 0.08, "outdoor": 0.5}
MIN_POINTS = 20
MIN_SPAN = 2.0
COLORS = {"d435i/mono": "#dc6947", "d435i/stereo": "#326fa6",
          "cuav-45deg/mono": "#a87332", "cuav-45deg/stereo": "#389377",
          "cuav-forward/mono": "#a87332", "cuav-forward/stereo": "#389377"}
METRICS = ("n", "span_s", "ate_rmse_m", "ate_median_m", "ate_p95_m",
           "rpe_1s_rmse_m", "rpe_1s_median_m", "rpe_1s_p95_m", "rpe_1s_n",
           "rpe_5s_rmse_m", "rpe_5s_median_m", "rpe_5s_p95_m", "rpe_5s_n",
            "sim3_scale", "scale_error_pct", "sim3_ate_rmse_m", "endpoint_drift_pct")


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def fmt(value, digits=2):
    x = finite(value)
    return f"{x:.{digits}f}" if math.isfinite(x) else "—"


def trajectory(path, tum=False):
    if tum:
        data = np.loadtxt(path, usecols=(0, 1, 2, 3), ndmin=2)
    else:
        # Ignore the trailing empty CSV field, quaternion and velocity.
        with path.open(newline="", encoding="utf-8") as stream:
            data = np.asarray([[float(v) for v in row[:4]] for row in csv.reader(stream)
                               if len(row) >= 4], dtype=float).reshape(-1, 4)
    if len(data) < 2 or not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError(f"时间戳不单调、点数不足或数值无效: {path}")
    return data


def interpolate(data, times, max_gap):
    """Return interpolated positions and validity, never bridge missing data."""
    t = data[:, 0]
    hi = np.searchsorted(t, times, side="left")
    exact = (hi < len(t)) & (t[np.minimum(hi, len(t) - 1)] == times)
    lo = np.clip(hi - 1, 0, len(t) - 1)
    upper = np.clip(hi, 0, len(t) - 1)
    dt = t[upper] - t[lo]
    valid = exact | ((hi > 0) & (hi < len(t)) & (dt > 0) & (dt <= max_gap))
    weight = np.divide(times - t[lo], dt, out=np.zeros_like(times), where=dt > 0)
    pos = data[lo, 1:4] + weight[:, None] * (data[upper, 1:4] - data[lo, 1:4])
    pos[~valid] = np.nan
    return pos, valid


def jump_mask(gt):
    """Flag both ends of a fast UWB jump and their immediate neighbours."""
    dt = np.diff(gt[:, 0])
    jump = (dt > 0) & (dt <= 0.03) & (np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1) > 2.0)
    bad = np.zeros(len(gt), dtype=bool)
    for offset in (-1, 0, 1, 2):
        indices = np.flatnonzero(jump) + offset
        indices = indices[(indices >= 0) & (indices < len(gt))]
        bad[indices] = True
    return ~bad, int(jump.sum())


def rpe_endpoint_valid(trace, times, delay):
    future = times + delay
    _, valid = interpolate(trace, future, VIO_GAP)
    t = trace[:, 0]
    start = np.searchsorted(t, times, side="left")
    end = np.searchsorted(t, future, side="left")
    gap_count = np.r_[0, np.cumsum(np.diff(t) > VIO_GAP)]
    valid &= gap_count[np.minimum(end, len(t) - 1)] == gap_count[np.minimum(start, len(t) - 1)]
    return valid


def align(est, ref, sim=False):
    x, y = est - est.mean(axis=0), ref - ref.mean(axis=0)
    if np.sum(x * x) < 1e-12:
        raise ValueError("退化的 VIO 轨迹")
    u, _, vt = np.linalg.svd(x.T @ y)
    diagonal = np.diag([1., 1., np.linalg.det(u @ vt)])
    rot = u @ diagonal @ vt
    scale = float(np.sum((x @ rot) * y) / np.sum(x * x)) if sim else 1.0
    return (est - est.mean(axis=0)) @ rot * scale + ref.mean(axis=0), scale, rot


def distribution(values):
    if not len(values):
        return (math.nan, math.nan, math.nan)
    return (float(np.sqrt(np.mean(values ** 2))), float(np.median(values)),
            float(np.percentile(values, 95)))


def score(gt, vio, trace, available, mask, gt_gap, shared_future=None):
    indices = np.flatnonzero(available & mask)
    if len(indices) < MIN_POINTS or gt[indices[-1], 0] - gt[indices[0], 0] < MIN_SPAN:
        return {}, None
    times = gt[indices, 0]
    ref = gt[indices, 1:4]
    est = vio[indices]
    try:
        aligned, _, rot = align(est, ref)
        sim_aligned, scale, _ = align(est, ref, sim=True)
    except ValueError:
        return {}, None
    # scale maps estimated positions into reference metres; its reciprocal is VIO/reference scale.
    reference_path = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    result = {"n": len(indices), "span_s": float(times[-1] - times[0]),
              "sim3_scale": scale,
              "scale_error_pct": abs(1.0 / scale - 1.0) * 100 if scale > 1e-12 else math.nan,
              "endpoint_drift_pct": (float(np.linalg.norm(aligned[-1] - ref[-1])) / reference_path * 100
                                     if reference_path > 1.0 else math.nan),
              "sim3_ate_rmse_m": distribution(np.linalg.norm(sim_aligned - ref, axis=1))[0]}
    result.update(zip(METRICS[2:5], distribution(np.linalg.norm(aligned - ref, axis=1))))
    # RPE uses the same fixed rotation as ATE; translations cancel and scale stays 1.
    # Guard every reference sample between endpoints, not just the two endpoints.
    bad_prefix = np.r_[0, np.cumsum(~mask)]
    gap_prefix = np.r_[0, np.cumsum(np.diff(gt[:, 0]) > gt_gap)]
    for delay in (1, 5):
        future = times + delay
        ref_future, ref_valid = interpolate(gt, future, gt_gap)
        end = np.searchsorted(gt[:, 0], future, side="left")
        interior_ok = bad_prefix[np.minimum(end + 1, len(gt))] == bad_prefix[indices]
        est_future, est_valid = interpolate(trace, future, VIO_GAP)
        start_index = np.searchsorted(gt[:, 0], times, side="left")
        uninterrupted = (gap_prefix[np.minimum(end, len(gt) - 1)] ==
                         gap_prefix[start_index])
        ok = (ref_valid & est_valid & interior_ok & uninterrupted &
              rpe_endpoint_valid(trace, times, delay) & (future <= gt[-1, 0]))
        if shared_future is not None:
            ok &= shared_future[delay][indices]
        delta = (est_future[ok] - est[ok]) @ rot - (ref_future[ok] - ref[ok])
        values = np.linalg.norm(delta, axis=1)
        result.update(zip((f"rpe_{delay}s_rmse_m", f"rpe_{delay}s_median_m",
                           f"rpe_{delay}s_p95_m"), distribution(values)))
        result[f"rpe_{delay}s_n"] = len(values)
    return result, aligned


def fields():
    base = ("sequence", "profile", "mode", "status", "reason", "run_dir", "reference_set",
            "cutoff_confidence",
            "bag_start", "evaluation_end", "gt_n", "gt_mask_n", "gt_jumps",
            "vio_n", "output_hz", "coverage", "matched_n", "common_n",
            "reference_coverage_pct", "common_coverage_pct",
            "drift_radius_median_m", "drift_radius_p95_m", "drift_radius_max_m", "path_length_m",
            "cpu_avg_pct", "rss_max_mib", "gpu_avg_pct", "solver_avg_ms", "solver_max_ms",
            "playback_factor", "notes")
    return list(base) + [f"{prefix}_{name}" for prefix in
                          ("own_raw", "common_raw", "own_masked", "common_masked") for name in METRICS]


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def figure(seq, records, gt, traces, output, start, reference_set):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax_xy, ax_err, ax_perf, ax_coverage = axes.flat
    if gt is not None:
        ax_xy.plot(gt[:, 1], gt[:, 2], color="#252e37", lw=2.2,
                   label="参考轨迹 GT/" + reference_set, zorder=5)
    for record, data in zip(records, traces):
        if data is None:
            continue
        label = record["profile"] + "/" + record["mode"]
        color = COLORS.get(label, "#996699")
        if gt is None:
            ax_xy.plot(data[:, 1] - data[0, 1], data[:, 2] - data[0, 2],
                       lw=1.2, color=color, label=label)
            radius = np.linalg.norm(data[:, 1:4] - data[0, 1:4], axis=1)
            ax_err.plot(data[:, 0] - start, radius, lw=1.2, color=color, label=label)
        else:
            aligned = record.get("_aligned")
            matched = record.get("_matched")
            if aligned is not None:
                ax_xy.plot(aligned[:, 0], aligned[:, 1], lw=1.2, color=color, label=label)
                ax_err.plot(gt[matched, 0] - start,
                            np.linalg.norm(aligned - gt[matched, 1:4], axis=1),
                            lw=1.1, color=color, label=label)
    ax_xy.set(xlabel="X (m)", ylabel="Y (m)", title="参考与 SE(3) 配准轨迹" if gt is not None
              else "相对初始位移（无位置真值）")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax_err.set(xlabel="从 bag 开始时间 (s)", ylabel="位置误差 (m)" if gt is not None
               else "距初始估计位移 (m)", title="位置误差时间序列（参考非绝对真值）" if gt is not None
               else "估计位移时间序列（不代表实际误差）")
    if gt is not None:
        ax_err.set_yscale("symlog", linthresh=1)
    ax_err.grid(alpha=0.2)
    labels = [r["profile"].replace("cuav-", "c-") + "/" + r["mode"][0] for r in records]
    x = np.arange(len(records))
    for i, r in enumerate(records):
        if r["status"] == "SUCCESS":
            ax_perf.bar(i - .18, finite(r.get("cpu_avg_pct")), .35, color="#dc6947")
            ax_perf.bar(i + .18, finite(r.get("solver_avg_ms")), .35, color="#326fa6")
        else:
            ax_perf.text(i, 0, r["status"], rotation=90, va="bottom", fontsize=8)
    ax_perf.set(title="计算开销", ylabel="CPU 平均占用 (%) / 求解均值 (ms)")
    ax_perf.plot([], [], color="#dc6947", lw=6, label="CPU %")
    ax_perf.plot([], [], color="#326fa6", lw=6, label="求解 ms")
    ax_perf.legend(fontsize=8)
    ax_coverage.bar(x, [finite(r.get("coverage")) for r in records], color="#389377")
    ax_coverage.axhline(1, color="black", lw=.8, ls="--")
    ax_coverage.set(title="轨迹时间跨度 / bag 时长", ylabel="覆盖比值", ylim=(0, 1.25))
    for ax in (ax_perf, ax_coverage):
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    policy = ("GT/full 人工去尾" if reference_set == "full" else "GT/eval 作者原始短区间")
    fig.suptitle(seq + "  |  " + policy + "；室内 UWB 精度仅供探索", fontsize=14)
    suffix = "" if reference_set == "full" else "_author_eval"
    fig.savefig(output / "figures/comparisons" / (seq + suffix + ".png"), dpi=145)
    plt.close(fig)


def overview(records, output):
    groups = ["d435i/mono", "d435i/stereo", "cuav/mono", "cuav/stereo"]
    def group(r):
        return ("d435i" if r["profile"] == "d435i" else "cuav") + "/" + r["mode"]
    for name, key, ylabel, title in (
        ("accuracy", "common_raw_ate_rmse_m", "SE(3) ATE RMSE (m)", "室外 GNSS 同序列共同时间区间"),
        ("coverage", "coverage", "轨迹跨度 / bag 时长", "所有有轨迹组合的时间覆盖"),
        ("efficiency", "cpu_avg_pct", "估计器 CPU 平均占用 (%)", "所有有轨迹组合的 CPU 与求解时间"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for index, label in enumerate(groups):
            chosen = [r for r in records if r["status"] == "SUCCESS" and group(r) == label
                      and (name != "accuracy" or r["sequence"].startswith("Outdoor_"))]
            values = np.array([finite(r.get(key)) for r in chosen])
            ax.scatter(index + np.linspace(-.12, .12, len(values)), values,
                       color=COLORS.get(label.replace("cuav", "cuav-forward"), "#326fa6"), s=22, alpha=.7)
            good = values[np.isfinite(values)]
            if len(good):
                ax.plot(index, np.median(good), marker="_", color="black", ms=19, mew=3)
        ax.set(xticks=range(len(groups)), xticklabels=groups, ylabel=ylabel, title=title)
        if name == "accuracy":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        if name == "efficiency":
            other = ax.twinx()
            for i, label in enumerate(groups):
                vals = [finite(r.get("solver_avg_ms")) for r in records
                        if r["status"] == "SUCCESS" and group(r) == label]
                if vals:
                    other.plot(i, np.nanmedian(vals), "D", color="#a87332", ms=7)
            other.set_ylabel("求解时间组中位数 (ms，菱形)")
        fig.savefig(output / "figures" / (name + ".png"), dpi=145)
        plt.close(fig)


def evaluate_set(rows, sequences, reference_set, output, plot=False):
    results = []
    for seq in sorted({r["sequence"] for r in rows}):
        jobs = [r for r in rows if r["sequence"] == seq]
        info = sequences[seq]
        meta = yaml.safe_load((DATA / seq / "metadata.yaml").read_text(encoding="utf-8"))
        bag = meta["rosbag2_bagfile_information"]
        start = bag["starting_time"]["nanoseconds_since_epoch"] / 1e9
        duration = bag["duration"]["nanoseconds"] / 1e9
        end = start + (min(duration, float(info.get("evaluation_end_offset_s", duration)))
                       if reference_set == "full" else duration)
        ref_path = DATA / "GT" / reference_set / (seq + ".tum")
        gt = None
        mask = None
        jumps = 0
        unscorable_uwb = False
        if info["dynamic"] and ref_path.is_file():
            try:
                data = trajectory(ref_path, tum=True)
                gt = data[(data[:, 0] >= start) & (data[:, 0] <= end)]
                if len(gt) < 2:
                    gt = None
                elif info["scene"] == "indoor":
                    mask, jumps = jump_mask(gt)
                    unscorable_uwb = (float(np.max(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1))) > 1000
                                      or (1.0 - mask.mean()) > 0.25)
                else:
                    mask = np.ones(len(gt), dtype=bool)
            except (ValueError, OSError) as exc:
                print(f"{seq} {reference_set}: {exc}")
        records, traces, sampled, validity = [], [], [], []
        for job in jobs:
            run = Path(job["run_dir"]) if job["run_dir"] else None
            record = {"sequence": seq, "profile": job["profile"], "mode": job["mode"],
                      "status": job["status"], "reason": job["reason"], "run_dir": job["run_dir"],
                      "reference_set": reference_set, "bag_start": start, "evaluation_end": end,
                      "cutoff_confidence": info.get("cutoff_confidence", "") if reference_set == "full" else "",
                      "gt_n": len(gt) if gt is not None else 0,
                      "gt_mask_n": int(mask.sum()) if mask is not None else 0,
                      "gt_jumps": jumps, "notes": job["flags"]}
            trace = None
            if run and (run / "summary.csv").is_file():
                with (run / "summary.csv").open(newline="", encoding="utf-8") as stream:
                    summary = {r["metric"]: r["value"] for r in csv.DictReader(stream)}
                for field, metric, divisor in (
                    ("cpu_avg_pct", "estimator_cpu_average", 1),
                    ("rss_max_mib", "estimator_rss_maximum", 1024),
                    ("gpu_avg_pct", "gpu_utilization_average", 1),
                    ("solver_avg_ms", "solver_average", 1),
                    ("solver_max_ms", "solver_maximum", 1)):
                    record[field] = finite(summary.get(metric)) / divisor
                wall = finite(summary.get("playback_wall"))
                record["playback_factor"] = duration / wall if wall > 0 else math.nan
            if job["status"] == "SUCCESS" and run:
                try:
                    trace = trajectory(run / "vio.csv")
                    span = trace[-1, 0] - trace[0, 0]
                    record.update(vio_n=len(trace), output_hz=(len(trace) - 1) / span,
                                  coverage=span / duration)
                    if not info["dynamic"]:
                        delta = trace[:, 1:4] - trace[0, 1:4]
                        radius = np.linalg.norm(delta, axis=1)
                        record.update(drift_radius_median_m=float(np.median(radius)),
                                      drift_radius_p95_m=float(np.percentile(radius, 95)),
                                      drift_radius_max_m=float(radius.max()),
                                      path_length_m=float(np.linalg.norm(np.diff(trace[:, 1:4], axis=0), axis=1).sum()))
                except (ValueError, OSError) as exc:
                    record["notes"] = (record["notes"] + "; " + str(exc)).strip("; ")
            if gt is not None and trace is not None:
                pos, valid = interpolate(trace, gt[:, 0], VIO_GAP)
                record["matched_n"] = int(valid.sum())
                record["reference_coverage_pct"] = float(valid.mean() * 100)
            else:
                pos, valid = None, None
            records.append(record)
            traces.append(trace)
            sampled.append(pos)
            validity.append(valid)
        usable = [v for v in validity if v is not None]
        common = np.logical_and.reduce(usable) if usable else None
        shared_future = ({delay: np.logical_and.reduce(
            [rpe_endpoint_valid(trace, gt[:, 0], delay) for trace in traces if trace is not None])
                          for delay in (1, 5)} if gt is not None and usable else None)
        for r, pos, trace, valid in zip(records, sampled, traces, validity):
            if gt is None or pos is None:
                if info["dynamic"]:
                    r["notes"] = (r["notes"] + "; 无可用GT或VIO").strip("; ")
                continue
            r["common_n"] = int(common.sum())
            r["common_coverage_pct"] = float(common.mean() * 100)
            for prefix, selection in (("own_raw", valid), ("common_raw", common),
                                       ("own_masked", valid), ("common_masked", common)):
                if "masked" in prefix and info["scene"] != "indoor":
                    continue
                if "masked" in prefix and mask.sum() < len(mask) / 2:
                    continue
                if "masked" in prefix and unscorable_uwb:
                    continue
                chosen = mask if "masked" in prefix else np.ones(len(gt), bool)
                metrics, aligned = score(gt, pos, trace, selection, chosen, GT_GAP[info["scene"]],
                                         shared_future if prefix.startswith("common") else None)
                r.update({f"{prefix}_{key}": val for key, val in metrics.items()})
                if prefix == "own_raw" and aligned is not None:
                    r["_aligned"] = aligned
                    r["_matched"] = np.flatnonzero(selection & chosen)
            if not r.get("own_raw_n"):
                r["notes"] = (r["notes"] + "; 无重叠或有效样本不足").strip("; ")
            if info["scene"] == "indoor" and not r.get("own_masked_n"):
                r["notes"] = (r["notes"] + "; UWB掩码后样本不足，不可评分").strip("; ")
            if info["scene"] == "indoor" and mask.sum() < len(mask) / 2:
                r["notes"] = (r["notes"] + "; UWB可疑点过多，不可评分").strip("; ")
            if info["scene"] == "indoor" and unscorable_uwb:
                r["notes"] = (r["notes"] + "; UWB明显大幅跳变，整段不具可信精度参考").strip("; ")
        if plot:
            figure(seq, records, gt, traces, output, start, reference_set)
        results.extend(records)
    return results


def markdown(rows, full, supplementary, output):
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    lines = ["# FWAF-VID v0 位置参考评估", "",
             "仅离线读取既有 manifest 与运行记录，不启动 ROS；下列所有 ATE/RPE 仅为位置指标。",
             f"状态：{', '.join(f'{k} {v}' for k, v in counts.items())}；"
             f"GT/full 有效原始对齐 {sum(bool(r.get('own_raw_n')) for r in full)} 组，"
             f"共享时刻对齐 {sum(bool(r.get('common_raw_n')) for r in full)} 组；"
              f"GT/eval 有效原始对齐 {sum(bool(r.get('own_raw_n')) for r in supplementary)} 组。", "",
              "## 快速阅读：同一室外序列的共同时间比较", "",
              "下表固定每条序列的同一组参考时间戳，数值为 SE(3) 位置 ATE RMSE (m)，"
              "不能与原先按各自区间计算的 ATE 混比；UWB 未列入排名。", "",
              "| 序列 | D435i 单目 | D435i 双目 | CUAV 单目 | CUAV 双目 | 共同参考覆盖 % |",
              "|---|---:|---:|---:|---:|---:|"]
    for seq in sorted({r["sequence"] for r in full if r["sequence"].startswith("Outdoor_")}):
        entries = [r for r in full if r["sequence"] == seq]
        def by_mode(sensor, mode):
            match = next((r for r in entries if (r["profile"] == "d435i") == (sensor == "d435i")
                          and r["mode"] == mode), None)
            return fmt(match.get("common_raw_ate_rmse_m")) if match else "—"
        lines.append("| " + " | ".join((seq, by_mode("d435i", "mono"), by_mode("d435i", "stereo"),
                                       by_mode("cuav", "mono"), by_mode("cuav", "stereo"),
                                       fmt(entries[0].get("common_coverage_pct"), 1))) + " |")
    lines += ["", f"**关键提醒**：{counts.get('SUCCESS', 0)} 组有轨迹不等于同样多的准确定位。"
              "室内参考跳变、单目迟初始化及大尺度偏差均需连同覆盖率和失败记录查看。", "",
              "## 方法与边界", "",
              "主参考 `data/ROS2/FWAF-VID/GT/full/<seq>.tum`，以 metadata.yaml 的 bag 起点"
              "和 config/FWAF-VID/evaluation_intervals.yaml 人工核验的 evaluation_end_offset_s 截断；"
              "独立的 `GT/eval` 是作者选定的更短区间，使用其原有完整时间范围"
              "（不再施加人工着陆阈值），见 [单独CSV](accuracy_eval.csv)；"
              "不与 full 拼接、混合或汇总排名。静态序列无位置 GT。", "",
              "GT 由参考话题的 bag 记录时间标记，VIO 使用图像时间戳；没有独立校准"
              "GNSS/UWB 到相机的时间偏移，故短基线 RPE 对残余时差敏感。", "",
             "以参考时间戳为网格，线性插值 VIO (最大间隔 0.25 s)；1/5 s 位移误差还插值终点，"
             "并限制 GT 间隔 (室内 0.08 s，室外 0.5 s)。每条轨迹在自身重叠区独立拟合固定尺度"
             " SE(3) 刚体旋转和平移；同序列 SUCCESS 且轨迹可读的所有条件再取共同有效参考时刻，"
             "在共同时间戳重新拟合，列名 own/common 不可混比。少于 20 点或不足 2 s 留空；"
              "RPE 采用相同 SE(3) 旋转，对 t 到 t+1 / t+5 的位移差求范数，"
              "不跨无效间隔。Sim(3) 拟合比例 s 是 VIO→参考的校正因子；尺度误差定义为"
              "|1/s-1|×100%。端点漂移百分比是 SE(3) 对齐后终点误差/匹配参考路程×100%。"
              "两者仅诊断尺度和累计漂移，Sim(3) 不用于主 ATE。", "",
             "室内 UWB 有严重跳变，**不具权威精度意义**。raw 仅供探索；masked 为 GT-only"
             " 规则：相邻时间差不大于 0.03 s 且跳变超过 2 m，删除两端及各一邻点，"
             "并禁止 RPE 跨被剔除点。阈值不能清除缓慢错误或长间隔异常；"
              "可用点少时标记不可评分；超过 1000 m 的 GT 瞬时跳变或剔除超过 25% 点则"
              "拒绝给出室内掩码后精度指标。室外 GNSS 非 RTK fixed，且无相机/IMU 与 GNSS"
             " 杆臂标定，结果是参考一致性而非绝对真值精度。未评估姿态。", "",
              "这批旧运行在上一轮采用外部文件读取原始 Kalibr/Allan；当前六份内联配置抄录了"
              "相同数值，但旧轨迹并非由新配置文件重新运行得到。", "",
              "覆盖率为 VIO 首末时间跨度 / bag 时长，不能证明中间无中断；Hz=(点数-1)/跨度；"
             "播放实时倍数=bag 时长/playback_wall；CPU/RSS/GPU/solver 源于原 summary.csv，"
             "RSS KB/1024 为 MiB，GPU 是采样利用率非本进程占用。静态半径与路程仅由 VIO"
             " 估计位置计算，不能断言真实漂移。空白表示不可用而不是零。", "",
             "总览精度纵轴为对数尺度；分序列误差曲线使用对称对数尺度以保留低误差细节。", "",
             "## 总览", "", "![室外共同区间精度](figures/accuracy.png)", "",
             "![轨迹时间覆盖](figures/coverage.png)", "",
             "![CPU 与求解时间](figures/efficiency.png)", "",
             "## 各序列（GT/full）", ""]
    for seq in sorted({r["sequence"] for r in full}):
        selected = [r for r in full if r["sequence"] == seq]
        static = "flapping" in seq
        indoor = seq.startswith("Indoor")
        lines += [f"### {seq}", "",
                   f"参考：{'无 GT，只有估计位移' if static else 'UWB（探索性）' if indoor else '非RTK GNSS'}；"
                   f"参考点 {selected[0]['gt_n']}，掩码后 {selected[0]['gt_mask_n']}，"
                   f"突变 {selected[0]['gt_jumps']}；裁剪终点相对 bag 起点 "
                   f"{fmt(selected[0]['evaluation_end'] - selected[0]['bag_start'], 1)} s"
                   f"（人工截断置信度：{selected[0]['cutoff_confidence'] or '不适用'}）。", ""]
        if static:
            lines += ["无参考位置，以下半径不是 ATE。", "",
                      "| 条件 | 状态 | 覆盖 | 自身位移半径 p95 / 最大值 (m) | 估计路程 (m) |",
                      "|---|---|---:|---:|---:|"]
        else:
            lines += ["同一序列比较请优先看**共同时间戳** ATE；自身区间和共同区间并不一定等长。", "",
                      "| 条件 | 状态 | 参考匹配 自身/共同 (%) | ATE RMSE 自身/共同 (m) | "
                      "ATE 中位/p95 自身 (m) | RPE 1s/5s 自身 (m) | 尺度误差/端点漂移 (%) |",
                      "|---|---|---:|---:|---:|---:|---:|"]
        for r in selected:
            def pair(a, b):
                return fmt(r.get(a)) + " / " + fmt(r.get(b))
            def counts(a, b):
                return fmt(r.get(a), 0) + " / " + fmt(r.get(b), 0)
            condition = r["profile"] + "/" + r["mode"]
            state = (r["status"] + " " + (r.get("reason") or r.get("notes") or "")).replace("|", "/")
            if static:
                values = (condition, state, fmt(r.get("coverage")),
                          pair("drift_radius_p95_m", "drift_radius_max_m"), fmt(r.get("path_length_m")))
            else:
                values = (condition, state,
                          pair("reference_coverage_pct", "common_coverage_pct"),
                          pair("own_raw_ate_rmse_m", "common_raw_ate_rmse_m"),
                          pair("own_raw_ate_median_m", "own_raw_ate_p95_m"),
                          pair("own_raw_rpe_1s_rmse_m", "own_raw_rpe_5s_rmse_m"),
                          pair("own_raw_scale_error_pct", "own_raw_endpoint_drift_pct"))
            lines.append("| " + " | ".join(values) + " |")
        if not static and indoor:
            lines += ["", "**UWB 仅供探索**：下表使用预声明 GT 跳点掩码，"
                      "若整段参考不可信则不显示掩码后误差。", "",
                      "| 条件 | 掩码后 ATE RMSE 自身/共同 (m) | UWB 跳变数 |",
                      "|---|---:|---:|"]
            for r in selected:
                lines.append("| " + " | ".join((r["profile"] + "/" + r["mode"],
                             fmt(r.get("own_masked_ate_rmse_m")) + " / "
                             + fmt(r.get("common_masked_ate_rmse_m")),
                             str(r["gt_jumps"]))) + " |")
        lines += ["", "运行效能（1.0x 播放，GPU 仅前端 OpenCV；CPU% 可大于 100 表示多核）：", "",
                  "| 条件 | CPU 均值 % | RSS 峰值 MiB | GPU 采样均值 % | 求解均值/最大 ms | 实时倍数 / 输出 Hz |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in selected:
            lines.append("| " + " | ".join((r["profile"] + "/" + r["mode"],
                         fmt(r.get("cpu_avg_pct")), fmt(r.get("rss_max_mib")),
                         fmt(r.get("gpu_avg_pct")), fmt(r.get("solver_avg_ms")) + " / "
                         + fmt(r.get("solver_max_ms")), fmt(r.get("playback_factor")) + " / "
                         + fmt(r.get("output_hz")))) + " |")
        lines += ["", f"![{seq} 轨迹、误差与性能](figures/comparisons/{seq}.png)", ""]
    lines += ["## 作者短区间（GT/eval，单独统计）", "",
              "[accuracy_eval.csv](accuracy_eval.csv) 仅包含存在 GT/eval 的 13 条序列；"
              "其原始区间与 full 的人工截断区间不相同，不能把两份统计当作相同真值比较。", "",
              "| 序列 | 条件 | 状态 | GT 点 | 自身/共同点 | raw ATE RMSE m | UWB 掩码 ATE RMSE m | 备注 |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in supplementary:
        lines.append("| " + " | ".join((r["sequence"], r["profile"] + "/" + r["mode"],
                     r["status"], str(r["gt_n"]),
                     fmt(r.get("own_raw_n"), 0) + " / " + fmt(r.get("common_raw_n"), 0),
                     fmt(r.get("own_raw_ate_rmse_m")), fmt(r.get("own_masked_ate_rmse_m")),
                      (r.get("reason") or r.get("notes") or "").replace("|", "/"))) + " |")
    lines += ["", "作者短区间逐序列参考与四组合轨迹对比："]
    for seq in sorted({r["sequence"] for r in supplementary}):
        lines += ["", f"#### {seq}", "",
                  f"![{seq} 作者短区间叠图](figures/comparisons/{seq}_author_eval.png)"]
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/v0_fwaf_full",
                        help="run directory containing manifest_status.csv; evaluation output goes here")
    args = parser.parse_args()
    with (args.output / "manifest_status.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sequences = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["sequences"]
    if {r["sequence"] for r in rows} != set(sequences):
        parser.error("manifest and sequence configuration disagree")
    output = args.output.resolve()
    (output / "figures/comparisons").mkdir(parents=True, exist_ok=True)
    full = evaluate_set(rows, sequences, "full", output, plot=True)
    eval_sequences = {p.stem for p in (DATA / "GT/eval").glob("*.tum")}
    short = evaluate_set([r for r in rows if r["sequence"] in eval_sequences],
                          sequences, "eval", output, plot=True)
    write_csv(output / "accuracy.csv", full)
    write_csv(output / "accuracy_eval.csv", short)
    overview(full, output)
    markdown(rows, full, short, output)
    print(f"output={output} full_rows={len(full)} eval_rows={len(short)} "
          f"figures={len(set(r['sequence'] for r in full)) + len(eval_sequences) + 3} "
          f"full_scored={sum(bool(r.get('own_raw_n')) for r in full)} "
          f"eval_scored={sum(bool(r.get('own_raw_n')) for r in short)}")


if __name__ == "__main__":
    main()
