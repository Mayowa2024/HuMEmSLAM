#!/usr/bin/env python3
"""Consolidate per-level KITTI darkness sweep analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ("ORB-SLAM3 baseline", "HuMemSLAM only")
COLORS = {"ORB-SLAM3 baseline": "#1f77b4", "HuMemSLAM only": "#2ca02c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    output = args.experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in range(10, 100, 10):
        source = (
            args.experiment / f"darkening_{level:02d}" /
            "analysis" / "aggregate_metrics.json"
        )
        data = json.loads(source.read_text())
        for mode in MODES:
            aggregate = data[mode]
            metrics = aggregate["metrics"]
            row = {
                "darkening_percent": level,
                "brightness_retained": 1.0 - level / 100.0,
                "mode": mode,
                "attempts": aggregate["attempts"],
                "complete_trajectories": aggregate["completed_trajectories"],
                "crashes": aggregate["crashes"],
                "loop_closures": aggregate["loop_closures"],
                "completion_rate": aggregate["completed_trajectories"] / aggregate["attempts"],
                "loop_closure_rate": aggregate["loop_closures"] / aggregate["attempts"],
            }
            for metric in (
                "ape_rmse_m", "current_frame", "matched_map_points",
                "correction_ms", "orb_track_mean_ms", "orb_track_p95_ms",
                "human_steady_mean_ms", "human_steady_p95_ms",
                "human_cold_start_ms",
            ):
                for statistic in ("n", "mean", "std"):
                    row[f"{metric}_{statistic}"] = metrics[metric][statistic]
            rows.append(row)

    fields = list(rows[0])
    with (output / "darkness_sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "darkness_sweep_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    def mode_rows(mode):
        return [row for row in rows if row["mode"] == mode]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODES:
        subset = mode_rows(mode)
        x = [row["darkening_percent"] for row in subset]
        y = [row["ape_rmse_m_mean"] for row in subset]
        e = [row["ape_rmse_m_std"] for row in subset]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=4,
                    linewidth=2, color=COLORS[mode], label=mode)
    ax.set(title="KITTI 06 accuracy under revisit darkening",
           xlabel="Darkening from frame 828 (%)", ylabel="Mean APE RMSE (m)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ape_rmse_vs_darkness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODES:
        subset = mode_rows(mode)
        ax.plot([row["darkening_percent"] for row in subset],
                [100 * row["loop_closure_rate"] for row in subset],
                marker="o", linewidth=2, color=COLORS[mode], label=mode)
    ax.set(title="KITTI 06 loop-closure success under revisit darkening",
           xlabel="Darkening from frame 828 (%)", ylabel="Runs with loop closure (%)",
           ylim=(-5, 105))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loop_closure_rate_vs_darkness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODES:
        subset = [row for row in mode_rows(mode) if row["current_frame_mean"] is not None]
        ax.errorbar([row["darkening_percent"] for row in subset],
                    [row["current_frame_mean"] for row in subset],
                    yerr=[row["current_frame_std"] for row in subset],
                    marker="o", capsize=4, linewidth=2,
                    color=COLORS[mode], label=mode)
    ax.set(title="KITTI 06 loop-closure detection frame",
           xlabel="Darkening from frame 828 (%)",
           ylabel="Mean closure frame (lower is earlier)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loop_closure_frame_vs_darkness.png", dpi=180)
    plt.close(fig)

    human = mode_rows("HuMemSLAM only")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([row["darkening_percent"] for row in human],
            [row["human_steady_mean_ms_mean"] for row in human],
            marker="o", linewidth=2, color=COLORS["HuMemSLAM only"])
    ax.set(title="HuMemSLAM steady-state latency under revisit darkening",
           xlabel="Darkening from frame 828 (%)",
           ylabel="Mean semantic inference latency (ms)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "humanslam_latency_vs_darkness.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
