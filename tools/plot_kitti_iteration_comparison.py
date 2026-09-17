#!/usr/bin/env python3
"""Create reproducible KITTI trajectory and frame-local error comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def poses(path):
    values = np.loadtxt(path, dtype=float).reshape(-1, 3, 4)
    out = np.repeat(np.eye(4)[None, :, :], len(values), axis=0)
    out[:, :3, :4] = values
    return out


def frame_ids(path):
    with open(path, newline="") as handle:
        return np.array([int(row["dataset_frame_id"]) for row in csv.DictReader(handle)])


def first_pose_error(est, gt):
    """Translation error after anchoring both paths at their own first pose."""
    est_rel = np.einsum("ij,nj->ni", est[0, :3, :3].T, est[:, :3, 3] - est[0, :3, 3])
    gt_rel = np.einsum("ij,nj->ni", gt[0, :3, :3].T, gt[:, :3, 3] - gt[0, :3, 3])
    return np.linalg.norm(est_rel - gt_rel, axis=1)


def align_se3(est_xyz, gt_xyz):
    src_mean, dst_mean = est_xyz.mean(0), gt_xyz.mean(0)
    u, _, vt = np.linalg.svd((est_xyz - src_mean).T @ (gt_xyz - dst_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (rotation @ est_xyz.T).T + dst_mean - rotation @ src_mean


def moving_mean(values, window=25):
    if len(values) < window:
        return values
    left = window // 2
    padded = np.pad(values, (left, window - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", required=True, type=Path)
    parser.add_argument("--perturb-start", type=int, default=835)
    parser.add_argument("--perturb-end", type=int, default=1100)
    args = parser.parse_args()

    specs = {
        "normal_orbslam3": ("normal/orbslam3_baseline", "Normal ORB-SLAM3", "#1f77b4"),
        "normal_humanslam": ("normal/humanslam", "Normal HuMemSLAM", "#2ca02c"),
        "perturbed_orbslam3": ("perturbed/orbslam3_baseline", "Perturbed ORB-SLAM3", "#d62728"),
        "perturbed_humanslam": ("perturbed/humanslam", "Perturbed HuMemSLAM", "#9467bd"),
    }
    plot_dir, metric_dir = args.iteration / "plots", args.iteration / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    data, summary = {}, {}
    for key, (relative, label, color) in specs.items():
        run = args.iteration / "runs" / relative
        est, gt, frames = poses(run / "trajectory_kitti.txt"), poses(run / "groundtruth_matched_kitti.txt"), frame_ids(run / "trajectory_frame_ids.csv")
        n = min(len(est), len(gt), len(frames))
        est, gt, frames = est[:n], gt[:n], frames[:n]
        local_error = first_pose_error(est, gt)
        aligned = align_se3(est[:, :3, 3], gt[:, :3, 3])
        ape = np.linalg.norm(aligned - gt[:, :3, 3], axis=1)
        data[key] = dict(est=est, gt=gt, frames=frames, local_error=local_error, ape=ape, aligned=aligned, label=label, color=color)
        before = local_error[frames < args.perturb_start]
        after = local_error[frames >= args.perturb_start]
        summary[key] = {
            "trajectory_poses": int(n),
            "first_pose_error_rmse_m": float(np.sqrt(np.mean(local_error ** 2))),
            "first_pose_error_pre_perturb_rmse_m": float(np.sqrt(np.mean(before ** 2))) if len(before) else None,
            "first_pose_error_post_perturb_rmse_m": float(np.sqrt(np.mean(after ** 2))) if len(after) else None,
            "ape_se3_rmse_m": float(np.sqrt(np.mean(ape ** 2))),
            "ape_se3_mean_m": float(np.mean(ape)),
            "ape_se3_median_m": float(np.median(ape)),
            "ape_se3_max_m": float(np.max(ape)),
        }

    max_frame = max(int(item["frames"].max()) for item in data.values())
    with open(metric_dir / "error_vs_frame_first_pose.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_frame_id", *specs])
        maps = {key: dict(zip(item["frames"], item["local_error"])) for key, item in data.items()}
        for frame in range(max_frame + 1):
            writer.writerow([frame, *[maps[key].get(frame, "") for key in specs]])
    with open(metric_dir / "run_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(metric_dir / "ape_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "rmse_m", "mean_m", "median_m", "max_m"])
        for key, values in summary.items():
            writer.writerow([key, values["ape_se3_rmse_m"], values["ape_se3_mean_m"], values["ape_se3_median_m"], values["ape_se3_max_m"]])

    def error_plot(keys, name, title):
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.axvspan(args.perturb_start, args.perturb_end, color="#f0c36e", alpha=.22, label="Night perturbation")
        for key in keys:
            item = data[key]
            ax.plot(item["frames"], item["local_error"], color=item["color"], alpha=.13, linewidth=.8)
            ax.plot(item["frames"], moving_mean(item["local_error"]), color=item["color"], linewidth=2, label=item["label"] + " (25-frame mean)")
        ax.axvline(args.perturb_start, color="#7f6000", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="KITTI dataset frame", ylabel="Translation error from first-pose alignment (m)")
        ax.grid(alpha=.25)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=180)
        plt.close(fig)

    error_plot(list(specs), "kitti06_iteration02_error_vs_frame_first_pose.png", "KITTI 06 iteration 02: frame-local trajectory error")
    error_plot(["perturbed_orbslam3", "perturbed_humanslam"], "perturbed_orbslam3_vs_humanslam_error_vs_frame.png", "Perturbed KITTI 06: ORB-SLAM3 vs HuMemSLAM")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, key in zip(axes.flat, specs):
        item = data[key]
        ax.plot(item["gt"][:, 0, 3], item["gt"][:, 2, 3], color="black", linewidth=1.7, label="Ground truth")
        ax.plot(item["aligned"][:, 0], item["aligned"][:, 2], color=item["color"], linewidth=1.4, label=item["label"])
        ax.set_title(item["label"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set(xlabel="x (m)", ylabel="z (m)")
    fig.suptitle("KITTI 06 iteration 02 trajectories (SE(3) aligned, no scale)")
    fig.tight_layout()
    fig.savefig(plot_dir / "kitti06_iteration02_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes.flat, specs):
        item = data[key]
        ax.plot(item["frames"], item["ape"], color=item["color"], linewidth=1)
        ax.axvspan(args.perturb_start, args.perturb_end, color="#f0c36e", alpha=.18)
        ax.set_title(f'{item["label"]} — RMSE {summary[key]["ape_se3_rmse_m"]:.3f} m')
        ax.grid(alpha=.2)
        ax.set(xlabel="KITTI dataset frame", ylabel="APE translation (m)")
    fig.tight_layout()
    fig.savefig(plot_dir / "kitti06_iteration02_ape_vs_frame.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
