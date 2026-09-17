#!/usr/bin/env python3
"""Plot a paired KITTI run without inventing APE for fragmented maps."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_FRAME = re.compile(r"dataset_frame=(\d+)")


def poses(path):
    values = np.loadtxt(path).reshape(-1, 3, 4)
    result = np.repeat(np.eye(4)[None], len(values), axis=0)
    result[:, :3, :4] = values
    return result


def index_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    frames, maps = [], []
    for row in rows:
        match = DATASET_FRAME.search(row.get("source_frame_id", ""))
        frames.append(int(match.group(1)) if match else int(row["frame_id"]))
        maps.append(int(row["map_id"]))
    return np.asarray(frames), np.asarray(maps)


def align(est, gt):
    source, target = est.mean(0), gt.mean(0)
    u, _, vt = np.linalg.svd((est - source).T @ (gt - target))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (rotation @ (est - source).T).T + target


def moving_mean(values, window=25):
    if len(values) < window:
        return values
    left = window // 2
    padded = np.pad(values, (left, window - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_run(run, gt_all):
    estimate = poses(run / "trajectory_kitti.txt")
    frames, maps = index_rows(run / "trajectory_frame_ids.csv")
    count = min(len(estimate), len(frames))
    estimate, frames, maps = estimate[:count], frames[:count], maps[:count]
    valid = (frames >= 0) & (frames < len(gt_all))
    estimate, frames, maps = estimate[valid], frames[valid], maps[valid]
    duplicate_count = len(frames) - len(set(frames.tolist()))
    map_count = len(set(maps.tolist()))
    global_valid = duplicate_count == 0 and map_count == 1
    aligned = error = None
    if global_valid:
        gt = gt_all[frames, :3, 3]
        aligned = align(estimate[:, :3, 3], gt)
        error = np.linalg.norm(aligned - gt, axis=1)
    return {
        "estimate": estimate, "frames": frames, "maps": maps,
        "duplicate_count": duplicate_count, "map_count": map_count,
        "global_valid": global_valid, "aligned": aligned, "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--perturb-start", type=int, default=828)
    args = parser.parse_args()

    gt_all = poses(args.ground_truth)
    output = args.experiment / "plots"
    output.mkdir(parents=True, exist_ok=True)
    runs = {
        "baseline": load_run(args.experiment / "baseline", gt_all),
        "humanslam": load_run(args.experiment / "humanslam", gt_all),
    }
    summary = {}
    for name, item in runs.items():
        summary[name] = {
            "trajectory_rows": int(len(item["frames"])),
            "unique_dataset_frames": int(len(set(item["frames"].tolist()))),
            "duplicate_dataset_associations": int(item["duplicate_count"]),
            "map_count": int(item["map_count"]),
            "global_ape_valid": bool(item["global_valid"]),
        }
        if item["global_valid"]:
            error = item["error"]
            summary[name]["ape_rmse_m"] = float(np.sqrt(np.mean(error ** 2)))
            summary[name]["ape_mean_m"] = float(np.mean(error))
            summary[name]["ape_median_m"] = float(np.median(error))
            summary[name]["ape_max_m"] = float(np.max(error))

    human = runs["humanslam"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ids = human["frames"]
    ax.plot(gt_all[ids, 0, 3], gt_all[ids, 2, 3], "k", lw=2, label="Ground truth")
    ax.plot(human["aligned"][:, 0], human["aligned"][:, 2], color="#2ca02c",
            lw=1.5, label="HuMemSLAM")
    ax.scatter(human["aligned"][ids == 916, 0], human["aligned"][ids == 916, 2],
               color="#d62728", s=45, zorder=4, label="Semantic closure (frame 916)")
    ax.set(title="HuMemSLAM trajectory — KITTI 06 blur 15 + dark 50",
           xlabel="x (m)", ylabel="z (m)")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "humanslam_trajectory.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(args.perturb_start, 1100, color="#f0c36e", alpha=.22,
               label="Blur 15 + dark 50")
    ax.plot(ids, human["error"], color="#2ca02c", alpha=.28, lw=.8)
    ax.plot(ids, moving_mean(human["error"]), color="#167c2d", lw=2,
            label="25-frame mean")
    ax.axvline(916, color="#d62728", ls="--", lw=1.3,
               label="Semantic loop correction")
    ax.set(title=f'HuMemSLAM translation APE — RMSE {summary["humanslam"]["ape_rmse_m"]:.3f} m',
           xlabel="KITTI dataset frame", ylabel="SE(3)-aligned translation error (m)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "humanslam_ape_vs_frame.png", dpi=200); plt.close(fig)

    base = runs["baseline"]
    if base["global_valid"]:
        base_ids = base["frames"]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(gt_all[base_ids, 0, 3], gt_all[base_ids, 2, 3], "k", lw=2,
                label="Ground truth")
        ax.plot(base["aligned"][:, 0], base["aligned"][:, 2], color="#1f77b4",
                lw=1.5, label="ORB-SLAM3 baseline")
        ax.set(title="ORB-SLAM3 baseline trajectory — KITTI 06 blur 15 + dark 50",
               xlabel="x (m)", ylabel="z (m)")
        ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output / "baseline_trajectory.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axvspan(args.perturb_start, 1100, color="#f0c36e", alpha=.22,
                   label="Blur 15 + dark 50")
        ax.plot(base_ids, base["error"], color="#1f77b4", alpha=.28, lw=.8)
        ax.plot(base_ids, moving_mean(base["error"]), color="#145a8d", lw=2,
                label="25-frame mean")
        ax.set(title=f'ORB-SLAM3 baseline translation APE — RMSE {summary["baseline"]["ape_rmse_m"]:.3f} m',
               xlabel="KITTI dataset frame", ylabel="SE(3)-aligned translation error (m)")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output / "baseline_ape_vs_frame.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(gt_all[human["frames"], 0, 3], gt_all[human["frames"], 2, 3],
                "k", lw=2, label="Ground truth")
        ax.plot(base["aligned"][:, 0], base["aligned"][:, 2], color="#1f77b4",
                lw=1.2, label="ORB-SLAM3 baseline")
        ax.plot(human["aligned"][:, 0], human["aligned"][:, 2], color="#2ca02c",
                lw=1.2, label="HuMemSLAM")
        ax.set(title="Matched trajectory comparison — repetition 1",
               xlabel="x (m)", ylabel="z (m)")
        ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output / "trajectory_comparison.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axvspan(args.perturb_start, 1100, color="#f0c36e", alpha=.22,
                   label="Blur 15 + dark 50")
        ax.plot(base_ids, moving_mean(base["error"]), color="#1f77b4", lw=2,
                label="ORB-SLAM3 baseline")
        ax.plot(human["frames"], moving_mean(human["error"]), color="#2ca02c",
                lw=2, label="HuMemSLAM")
        ax.set(title="Matched translation APE — 25-frame means",
               xlabel="KITTI dataset frame", ylabel="SE(3)-aligned translation error (m)")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output / "ape_comparison.png", dpi=200); plt.close(fig)
    else:


        fig, ax = plt.subplots(figsize=(8, 7))
        for map_id in sorted(set(base["maps"].tolist())):
            xyz = base["estimate"][base["maps"] == map_id, :3, 3]
            if len(xyz):
                ax.plot(xyz[:, 0], xyz[:, 2], color="#1f77b4", alpha=.25, lw=.8)
        ax.set(title=f'ORB-SLAM3 baseline local-map fragments ({base["map_count"]} maps)',
               xlabel="local x (m)", ylabel="local z (m)")
        ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(output / "baseline_trajectory_fragments.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(base["frames"], base["maps"], s=4, alpha=.5, color="#1f77b4")
        ax.axvspan(args.perturb_start, 1100, color="#f0c36e", alpha=.22)
        ax.set(title="ORB-SLAM3 baseline: global APE invalid after repeated map resets",
               xlabel="KITTI dataset frame", ylabel="ORB atlas map ID")
        ax.text(.02, .96,
                f'{base["map_count"]} maps; {base["duplicate_count"]} duplicate frame associations. '
                "No defensible global APE can be plotted.",
                transform=ax.transAxes, va="top", color="#b22222",
                bbox=dict(facecolor="white", alpha=.85, edgecolor="#b22222"))
        ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(output / "baseline_ape_invalid_map_resets.png", dpi=200); plt.close(fig)

    with (output / "plot_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
