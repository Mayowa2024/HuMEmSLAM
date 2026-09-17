#!/usr/bin/env python3
"""Aggregate the final Business School campaign using the KITTI table schema."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from aggregate_garage_results import details, human_latency, loss_episodes


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "test_results/dissertation_final/4seasons/stereo_only_comparison_5x_20260820"
OUTPUT = CAMPAIGN / "business_school_aggregate_metrics.csv"
DATASETS = {
    "4S-BF": ("Business School fall", "business_fall", "recording_2020-10-08_09-30-57", 10741, 87.950),
    "4S-BW": ("Business School winter", "business_winter", "recording_2021-01-07_13-12-23", -1, 88.0),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.fmean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def main() -> None:
    rows = []
    for dataset_id, (label, prefix, recording, end_frame, bow_ms) in DATASETS.items():
        times = ROOT / "test_scenarios/4seasons_business_school" / recording / "times.txt"
        available_frames = sum(1 for line in times.open(encoding="utf-8") if line.strip())
        frame_count = available_frames if end_frame < 0 else min(available_frames, end_frame + 1)
        for method in ("baseline", "humanslam"):
            runs = []
            for run in sorted(CAMPAIGN.glob(f"{prefix}_run_*")):
                directory = run / method
                metrics_path = run / "metrics/summary.json"
                if not directory.is_dir() or not metrics_path.is_file():
                    continue
                metrics = read_json(metrics_path)["ORB-SLAM3" if method == "baseline"
                                                  else "ORB-SLAM3 + HuMemSLAM"]
                retrieval_path = directory / "loop_retrieval_metrics.json"
                retrieval = read_json(retrieval_path) if retrieval_path.is_file() else None
                attempts = accepted = closures = maps_created = maps_merged = 0
                with (directory / "orb_events.csv").open(newline="", encoding="utf-8") as stream:
                    for event in csv.DictReader(stream):
                        name = event.get("event")
                        if name == "RETRIEVAL_GEOMETRIC_RESULT":
                            attempts += 1
                            accepted += details(event.get("details", "")).get("accepted_seed") == "1"
                        elif name == "LOOP_CORRECTED": closures += 1
                        elif name == "MAP_CREATED": maps_created += 1
                        elif name == "MAP_MERGE_SUCCESS": maps_merged += 1
                losses, recovered = loss_episodes(directory / "orb_events.csv")
                runs.append({
                    "retrieval": retrieval, "attempts": attempts, "accepted": accepted,
                    "closures": closures, "completeness": metrics["matched_poses"] / frame_count,
                    "ape": metrics["ape_rmse_m"], "rpe": metrics["rpe_translation_rmse_m"],
                    "orb_ms": metrics["tracking_latency_mean_ms"],
                    "human_ms": human_latency(directory / "human_latency.csv"),
                    "losses": losses, "recovered": recovered,
                    "maps_created": maps_created, "maps_merged": maps_merged,
                })
            retrieval_runs = [row for row in runs if row["retrieval"] is not None]
            rows.append({
                "id": dataset_id, "dataset": label,
                "method": "ORB-SLAM3" if method == "baseline" else "HuMemSLAM",
                "trajectory_runs": len(runs), "retrieval_runs": len(retrieval_runs),
                "evaluable_trajectories": len(runs),
                "runs_with_closure": sum(row["closures"] > 0 for row in runs),
                "recall_at_1_mean": mean([row["retrieval"]["recall@1"] for row in retrieval_runs]),
                "recall_at_5_mean": mean([row["retrieval"]["recall@5"] for row in retrieval_runs]),
                "mrr_mean": mean([row["retrieval"]["mrr"] for row in retrieval_runs]),
                "geometry_accepted_seeds": sum(row["accepted"] for row in runs),
                "geometry_attempts": sum(row["attempts"] for row in runs),
                "applied_closures": sum(row["closures"] for row in runs),
                "completeness_mean": mean([row["completeness"] for row in runs]),
                "ape_rmse_mean_m": mean([row["ape"] for row in runs]),
                "ape_rmse_sd_m": sd([row["ape"] for row in runs]),
                "rpe_rmse_mean_m": mean([row["rpe"] for row in runs]),
                "rpe_rmse_sd_m": sd([row["rpe"] for row in runs]),
                "orb_tracking_mean_ms": mean([row["orb_ms"] for row in runs]),
                "native_bow_pipeline_estimated_ms": bow_ms if method == "baseline" else None,
                "human_steady_mean_ms": mean([row["human_ms"] for row in runs if row["human_ms"] is not None]),
                "loss_episodes": sum(row["losses"] for row in runs),
                "recovered_episodes": sum(row["recovered"] for row in runs),
                "maps_created": sum(row["maps_created"] for row in runs),
                "maps_merged": sum(row["maps_merged"] for row in runs),
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
