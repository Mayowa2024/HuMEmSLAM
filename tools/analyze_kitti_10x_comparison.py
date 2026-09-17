#!/usr/bin/env python3
"""Aggregate repeated KITTI ORB-SLAM3 versus HuMemSLAM-only experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOOP_PATTERN = re.compile(
    r"Loop closure metrics: source=(?P<source>[^;]+); "
    r"current_frame=(?P<current_frame>\d+); current_kf=(?P<current_kf>\d+); "
    r"matched_kf=(?P<matched_kf>\d+); matched_map_points=(?P<matched_map_points>\d+); "
    r"temporal_verification_ms=(?P<temporal_verification_ms>[-+\d.eE]+); "
    r"correction_ms=(?P<correction_ms>[-+\d.eE]+)"
)


def load_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float).reshape(-1, 3, 4)
    poses = np.repeat(np.eye(4)[None, :, :], len(values), axis=0)
    poses[:, :3, :4] = values
    return poses


def align_se3(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    src_mean = est_xyz.mean(axis=0)
    dst_mean = gt_xyz.mean(axis=0)
    u, _, vt = np.linalg.svd((est_xyz - src_mean).T @ (gt_xyz - dst_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (rotation @ est_xyz.T).T + dst_mean - rotation @ src_mean


def numeric_column(path: Path, column: str, event: str | None = None) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=float)
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if event is not None and row.get("event") != event:
                continue
            raw = row.get(column, "")
            if raw not in (None, ""):
                values.append(float(raw))
    return np.asarray(values, dtype=float)


def orb_latency_at_frame(path: Path, frame: int) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event") == "track" and int(row["frame_id"]) == frame:
                return float(row["track_stereo_ms"])
    return None


def stats(values: list[float | int | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def analyse_run(run_dir: Path, gt: np.ndarray, condition: str) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    record: dict[str, object] = {
        "condition": condition,
        "run": run_dir.name,
        "completed": False,
        "pose_count": 0,
        "crashed": False,
        "loop_closed": False,
    }
    console_path = run_dir / "run_console.log"
    console = console_path.read_text(errors="replace") if console_path.exists() else ""
    record["crashed"] = "process has died" in console or "SO3::exp failed" in console
    loop_match = LOOP_PATTERN.search(console)
    if loop_match:
        record["loop_closed"] = True
        for key, value in loop_match.groupdict().items():
            record[key] = value if key == "source" else float(value) if key.endswith("_ms") else int(value)

    trajectory_path = run_dir / "trajectory_kitti.txt"
    errors = None
    aligned = None
    if trajectory_path.exists() and trajectory_path.stat().st_size:
        est = load_poses(trajectory_path)
        record["pose_count"] = int(len(est))
        if len(est) == len(gt):
            record["completed"] = True
            aligned = align_se3(est[:, :3, 3], gt[:, :3, 3])
            errors = np.linalg.norm(aligned - gt[:, :3, 3], axis=1)
            record.update(
                ape_rmse_m=float(np.sqrt(np.mean(errors**2))),
                ape_mean_m=float(np.mean(errors)),
                ape_median_m=float(np.median(errors)),
                ape_std_m=float(np.std(errors)),
                ape_min_m=float(np.min(errors)),
                ape_max_m=float(np.max(errors)),
            )

    orb_track = numeric_column(run_dir / "orb_latency.csv", "track_stereo_ms", "track")
    if len(orb_track):
        record["orb_track_mean_ms"] = float(np.mean(orb_track))
        record["orb_track_p95_ms"] = float(np.percentile(orb_track, 95))
        record["orb_track_max_ms"] = float(np.max(orb_track))
    if loop_match:
        record["closure_frame_track_ms"] = orb_latency_at_frame(
            run_dir / "orb_latency.csv", int(record["current_frame"])
        )

    human_total = numeric_column(run_dir / "human_latency.csv", "total_ms")
    if len(human_total):
        record["human_total_mean_ms"] = float(np.mean(human_total))
        record["human_total_median_ms"] = float(np.median(human_total))
        record["human_total_p95_ms"] = float(np.percentile(human_total, 95))
        record["human_total_max_ms"] = float(np.max(human_total))



        steady = human_total[1:]
        if len(steady):
            record["human_steady_mean_ms"] = float(np.mean(steady))
            record["human_steady_median_ms"] = float(np.median(steady))
            record["human_steady_p95_ms"] = float(np.percentile(steady, 95))
            record["human_steady_max_ms"] = float(np.max(steady))
            record["human_cold_start_ms"] = float(human_total[0])
    return record, errors, aligned


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["condition", "run", "completed", "crashed", "loop_closed", "pose_count"]
    fields = preferred + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    args = parser.parse_args()

    output = args.experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    gt = load_poses(args.ground_truth)
    conditions = {
        "ORB-SLAM3 baseline": args.experiment / "baseline",
        "HuMemSLAM only": args.experiment / "humanslam_only",
    }

    records: list[dict] = []
    errors_by_condition: dict[str, list[np.ndarray]] = {key: [] for key in conditions}
    aligned_by_condition: dict[str, list[np.ndarray]] = {key: [] for key in conditions}
    for condition, directory in conditions.items():
        for run_dir in sorted(directory.glob("run_*")):
            record, errors, aligned = analyse_run(run_dir, gt, condition)
            records.append(record)
            if errors is not None:
                errors_by_condition[condition].append(errors)
                aligned_by_condition[condition].append(aligned)

    write_csv(output / "per_run_metrics.csv", records)
    aggregate: dict[str, dict] = {}
    metric_names = [
        "ape_rmse_m", "ape_mean_m", "ape_median_m", "ape_max_m",
        "current_frame", "current_kf", "matched_kf", "matched_map_points",
        "correction_ms", "closure_frame_track_ms", "orb_track_mean_ms",
        "orb_track_p95_ms", "human_total_mean_ms", "human_total_median_ms",
        "human_total_p95_ms", "human_cold_start_ms", "human_steady_mean_ms",
        "human_steady_median_ms", "human_steady_p95_ms", "human_steady_max_ms",
    ]
    for condition in conditions:
        subset = [row for row in records if row["condition"] == condition]
        aggregate[condition] = {
            "attempts": len(subset),
            "completed_trajectories": sum(bool(row["completed"]) for row in subset),
            "crashes": sum(bool(row["crashed"]) for row in subset),
            "loop_closures": sum(bool(row["loop_closed"]) for row in subset),
            "metrics": {
                metric: stats([row.get(metric) for row in subset]) for metric in metric_names
            },
        }
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))

    with (output / "aggregate_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "n", "mean", "std", "median", "min", "max"])
        for condition, values in aggregate.items():
            writer.writerow([condition, "completion_rate", values["attempts"], values["completed_trajectories"] / values["attempts"], "", "", "", ""])
            writer.writerow([condition, "loop_closure_rate", values["attempts"], values["loop_closures"] / values["attempts"], "", "", "", ""])
            for metric, metric_stats in values["metrics"].items():
                writer.writerow([condition, metric, *[metric_stats[key] for key in ("n", "mean", "std", "median", "min", "max")]])

    colors = {"ORB-SLAM3 baseline": "#1f77b4", "HuMemSLAM only": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 6))
    frames = np.arange(len(gt))
    for condition in conditions:
        matrix = np.vstack(errors_by_condition[condition])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        ax.plot(frames, mean, color=colors[condition], linewidth=2, label=f"{condition} mean")
        ax.fill_between(frames, mean - std, mean + std, color=colors[condition], alpha=0.18, label=f"{condition} ±1 SD")
    ax.set(title="KITTI 06 repeated runs: translation APE versus frame", xlabel="KITTI frame", ylabel="APE translation (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ape_vs_frame_mean_std.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ape_boxes = [[row["ape_rmse_m"] for row in records if row["condition"] == condition and "ape_rmse_m" in row] for condition in conditions]
    box = ax.boxplot(ape_boxes, tick_labels=list(conditions), patch_artist=True, showmeans=True)
    for patch, condition in zip(box["boxes"], conditions):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.55)
    ax.set(title="KITTI 06 APE RMSE across completed runs", ylabel="APE RMSE (m)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ape_rmse_boxplot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    closure_boxes = [[row["current_frame"] for row in records if row["condition"] == condition and row["loop_closed"]] for condition in conditions]
    box = ax.boxplot(closure_boxes, tick_labels=list(conditions), patch_artist=True, showmeans=True)
    for patch, condition in zip(box["boxes"], conditions):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.55)
    ax.set(title="Loop-closure detection frame across 10 attempts", ylabel="KITTI frame (lower is earlier)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loop_closure_frame_boxplot.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title, ylabel in (
        (axes[0], "closure_frame_track_ms", "ORB processing on loop-closure frame", "TrackStereo latency (ms)"),
        (axes[1], "correction_ms", "Loop graph-correction time", "Correction latency (ms)"),
    ):
        values = [[row[metric] for row in records if row["condition"] == condition and row.get(metric) is not None] for condition in conditions]
        box = ax.boxplot(values, tick_labels=list(conditions), patch_artist=True, showmeans=True)
        for patch, condition in zip(box["boxes"], conditions):
            patch.set_facecolor(colors[condition])
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loop_closure_latency_boxplots.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="black", linestyle="--", linewidth=2.2, label="Ground truth")
    for condition in conditions:
        for index, aligned in enumerate(aligned_by_condition[condition]):
            ax.plot(aligned[:, 0], aligned[:, 2], color=colors[condition], alpha=0.16, linewidth=1)
        mean_path = np.stack(aligned_by_condition[condition]).mean(axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 2], color=colors[condition], linewidth=2, label=f"{condition} mean")
    ax.set(title="KITTI 06 trajectories across repeated runs (SE(3) aligned, no scale)", xlabel="x (m)", ylabel="z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trajectory_comparison_all_runs.png", dpi=180)
    plt.close(fig)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
