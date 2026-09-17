#!/usr/bin/env python3
"""Aggregate the final multi-condition KITTI 06 campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def nested(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows if row.get(key) is not None])


def mean(rows, key):
    data = values(rows, key)
    return float(data.mean()) if len(data) else None


def sd(rows, key):
    data = values(rows, key)
    return float(data.std(ddof=1)) if len(data) > 1 else 0.0 if len(data) else None


def latency(path: Path):
    if not path.is_file():
        return None, None
    data = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("total_ms"):
                data.append(float(row["total_ms"]))
    if not data:
        return None, None
    steady = data[1:] if len(data) > 1 else data
    return float(np.mean(steady)), float(np.median(steady))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition_dir in sorted(args.root.glob("K06-*")):
        formal = condition_dir / "formal_10x"
        if not formal.is_dir():
            continue
        for run in sorted(formal.glob("run_*")):
            for mode in ("baseline", "humanslam"):
                directory = run / mode
                summary = read_json(directory / "run_summary.json")
                retrieval = read_json(directory / "loop_retrieval_metrics.json")
                relocal = read_json(directory / "relocalisation_metrics.json")
                source = "native_bow" if mode == "baseline" else "semantic"
                src = nested(retrieval, "sources", source, default={})
                ape_maps = nested(summary, "trajectory_evaluation", "maps", default={})
                ape = next(iter(ape_maps.values()), {}).get("ape", {}) if ape_maps else {}
                steady_mean, steady_median = latency(directory / "human_latency.csv")
                rows.append({
                    "condition": condition_dir.name,
                    "run": run.name,
                    "mode": mode,
                    "complete": nested(summary, "trajectory_evaluation", "tracking_completeness"),
                    "map_count": nested(summary, "trajectory_evaluation", "map_count"),
                    "ape_rmse_m": ape.get("rmse"),
                    "orb_mean_ms": nested(summary, "orb_tracking_ms", "mean"),
                    "recall1": src.get("recall@1"),
                    "recall5": src.get("recall@5"),
                    "mrr": src.get("mrr"),
                    "eligible_queries": src.get("eligible_revisit_queries"),
                    "geometry_attempts": nested(retrieval, "geometric_verification", "attempts"),
                    "geometry_accepted": nested(retrieval, "geometric_verification", "accepted_seeds"),
                    "closures": nested(retrieval, "loop_correction", "closures"),
                    "correction_ms": nested(retrieval, "loop_correction", "latency_ms_mean"),
                    "loss_episodes": nested(relocal, "tracking_loss", "episodes"),
                    "recovered_episodes": nested(relocal, "tracking_loss", "recovered_episodes"),
                    "relocal_successes": nested(relocal, "relocalisation", "successes"),
                    "human_steady_mean_ms": steady_mean,
                    "human_steady_median_ms": steady_median,
                })

    with (output / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    aggregate = []
    for condition in sorted({row["condition"] for row in rows}):
        for mode in ("baseline", "humanslam"):
            subset = [row for row in rows if row["condition"] == condition and row["mode"] == mode]
            aggregate.append({
                "condition": condition,
                "mode": mode,
                "runs": len(subset),



                "complete_runs": sum((row["complete"] or 0) >= 0.99 for row in subset),
                "ape_rmse_mean_m": mean(subset, "ape_rmse_m"),
                "ape_rmse_sd_m": sd(subset, "ape_rmse_m"),
                "recall1_mean": mean(subset, "recall1"),
                "recall5_mean": mean(subset, "recall5"),
                "mrr_mean": mean(subset, "mrr"),
                "geometry_attempts": int(sum(row["geometry_attempts"] or 0 for row in subset)),
                "geometry_accepted": int(sum(row["geometry_accepted"] or 0 for row in subset)),
                "runs_with_closure": sum((row["closures"] or 0) > 0 for row in subset),
                "closures": int(sum(row["closures"] or 0 for row in subset)),
                "tracking_loss_episodes": int(sum(row["loss_episodes"] or 0 for row in subset)),
                "recovered_episodes": int(sum(row["recovered_episodes"] or 0 for row in subset)),
                "relocal_successes": int(sum(row["relocal_successes"] or 0 for row in subset)),
                "orb_tracking_mean_ms": mean(subset, "orb_mean_ms"),
                "human_steady_mean_ms": mean(subset, "human_steady_mean_ms"),
            })
    with (output / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    labels = [x["condition"].replace("K06-", "") for x in aggregate[::2]]
    x = np.arange(len(labels)); width = .36
    base = aggregate[::2]; human = aggregate[1::2]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0,0].bar(x-width/2, [r["ape_rmse_mean_m"] for r in base], width, label="ORB-SLAM3")
    axes[0,0].bar(x+width/2, [r["ape_rmse_mean_m"] for r in human], width, label="HuMemSLAM")
    axes[0,0].set_ylabel("Mean APE RMSE (m)"); axes[0,0].legend()
    axes[0,1].bar(x-width/2, [r["runs_with_closure"] for r in base], width)
    axes[0,1].bar(x+width/2, [r["runs_with_closure"] for r in human], width)
    axes[0,1].set_ylabel("Runs with loop correction (/10)")
    axes[1,0].plot(x, [r["recall1_mean"] for r in base], "o-", label="BoW Recall@1")
    axes[1,0].plot(x, [r["recall1_mean"] for r in human], "o-", label="Semantic Recall@1")
    axes[1,0].set_ylabel("Ground-truth Recall@1"); axes[1,0].legend()
    axes[1,1].bar(x-width/2, [r["orb_tracking_mean_ms"] for r in base], width)
    axes[1,1].bar(x+width/2, [r["orb_tracking_mean_ms"] for r in human], width)
    axes[1,1].set_ylabel("Mean TrackStereo latency (ms)")
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("KITTI 06 final campaign: ORB-SLAM3 versus HuMemSLAM")
    fig.tight_layout(); fig.savefig(output / "campaign_summary.png", dpi=180); plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
