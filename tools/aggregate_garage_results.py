#!/usr/bin/env python3
"""Build a central KITTI-style results sheet for the final garage campaign."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "test_results/dissertation_final/4seasons/stereo_only_comparison_5x_20260820"
OUTPUT = CAMPAIGN / "garage_aggregate_metrics.csv"
DATASETS = {
    "4S-GD": ("Garage Dec 2020", "garage_dec2020", "recording_2020-12-22_12-04-35", 75.4),
    "4S-GF": ("Garage Feb 2021", "garage_feb2021", "recording_2021-02-25_13-39-06", 75.422),
    "4S-GM": ("Garage May 2021", "garage_may2021", "recording_2021-05-10_19-15-19", 75.4),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.fmean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def details(value: str) -> dict[str, str]:
    return {key.strip(): val.strip() for item in value.split(";") if "=" in item
            for key, val in (item.split("=", 1),)}


def human_latency(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        values = [float(row["total_ms"]) for row in csv.DictReader(stream)
                  if row.get("total_ms")]
    return mean(values[1:] if len(values) > 1 else values)


def loss_episodes(path: Path) -> tuple[int, int]:
    lost = {"RECENTLY_LOST", "LOST"}
    tracking = {"TRACKING_STATUS", "TRACKING_STABLE", "TRACKING_WEAK",
                "LOW_INLIERS", "TRACKING_RECENTLY_LOST", "TRACKING_LOST"}
    episodes = recovered = 0
    active = False
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("event") not in tracking:
                continue
            state = row.get("state_name", "")
            if state in lost and not active:
                episodes += 1
                active = True
            elif state == "OK" and active:
                recovered += 1
                active = False
    return episodes, recovered


def main() -> None:
    rows = []
    for dataset_id, (label, prefix, recording, bow_ms) in DATASETS.items():
        times = ROOT / "test_scenarios/4seasons_multilevel_carpark" / recording / "times.txt"
        frame_count = sum(1 for line in times.open(encoding="utf-8") if line.strip())
        for method in ("baseline", "humanslam"):
            per_run = []
            for run in sorted(CAMPAIGN.glob(f"{prefix}_run_*")):
                directory = run / method
                retrieval_path = directory / "loop_retrieval_metrics.json"
                alias_path = directory / "aliasing_evaluation/summary.json"
                metrics_path = run / "metrics/summary.json"
                if not all(path.is_file() for path in (retrieval_path, alias_path, metrics_path)):
                    continue
                retrieval = read_json(retrieval_path)
                alias = read_json(alias_path)
                accepted_overlap = false_accepted = 0
                with (directory / "aliasing_evaluation/events.csv").open(
                        newline="", encoding="utf-8") as stream:
                    for event in csv.DictReader(stream):
                        if event.get("accepted") != "1":
                            continue
                        if event.get("valid_place_overlap") == "1":
                            accepted_overlap += 1
                        else:
                            false_accepted += 1
                metrics = read_json(metrics_path)["ORB-SLAM3" if method == "baseline"
                                                  else "ORB-SLAM3 + HuMemSLAM"]
                attempts = accepted_seeds = closures = maps_created = maps_merged = 0
                with (directory / "orb_events.csv").open(newline="", encoding="utf-8") as stream:
                    for event in csv.DictReader(stream):
                        name = event.get("event")
                        if name == "RETRIEVAL_GEOMETRIC_RESULT":
                            attempts += 1
                            accepted_seeds += details(event.get("details", "")).get("accepted_seed") == "1"
                        elif name == "LOOP_CORRECTED":
                            closures += 1
                        elif name == "MAP_CREATED":
                            maps_created += 1
                        elif name == "MAP_MERGE_SUCCESS":
                            maps_merged += 1
                losses, recovered = loss_episodes(directory / "orb_events.csv")
                per_run.append({
                    "r1": retrieval.get("recall@1"), "r5": retrieval.get("recall@5"),
                    "mrr": retrieval.get("mrr"), "proposals": alias["loop_proposals"],
                    "accepted": alias["accepted"], "valid": accepted_overlap,
                    "false": false_accepted,
                    "attempts": attempts, "accepted_seeds": accepted_seeds,
                    "closures": closures, "complete": metrics["matched_poses"] / frame_count,
                    "ape": metrics["ape_rmse_m"], "rpe": metrics["rpe_translation_rmse_m"],
                    "orb_ms": metrics["tracking_latency_mean_ms"],
                    "human_ms": human_latency(directory / "human_latency.csv"),
                    "losses": losses, "recovered": recovered,
                    "maps_created": maps_created, "maps_merged": maps_merged,
                })
            rows.append({
                "id": dataset_id, "dataset": label, "method": "ORB-SLAM3" if method == "baseline" else "HuMemSLAM",
                "runs": len(per_run), "evaluable_trajectories": len(per_run),
                "runs_with_closure": sum(row["closures"] > 0 for row in per_run),
                "recall_at_1_mean": mean([row["r1"] for row in per_run]),
                "recall_at_5_mean": mean([row["r5"] for row in per_run]),
                "mrr_mean": mean([row["mrr"] for row in per_run]),
                "loop_proposals": sum(row["proposals"] for row in per_run),
                "accepted_proposals": sum(row["accepted"] for row in per_run),
                "valid_overlap_accepted": sum(row["valid"] for row in per_run),
                "false_accepted": sum(row["false"] for row in per_run),
                "geometry_accepted_seeds": sum(row["accepted_seeds"] for row in per_run),
                "geometry_attempts": sum(row["attempts"] for row in per_run),
                "applied_closures": sum(row["closures"] for row in per_run),
                "completeness_mean": mean([row["complete"] for row in per_run]),
                "ape_rmse_mean_m": mean([row["ape"] for row in per_run]),
                "ape_rmse_sd_m": sd([row["ape"] for row in per_run]),
                "rpe_rmse_mean_m": mean([row["rpe"] for row in per_run]),
                "rpe_rmse_sd_m": sd([row["rpe"] for row in per_run]),
                "orb_tracking_mean_ms": mean([row["orb_ms"] for row in per_run]),
                "native_bow_pipeline_estimated_ms": bow_ms if method == "baseline" else None,
                "human_steady_mean_ms": mean([row["human_ms"] for row in per_run if row["human_ms"] is not None]),
                "loss_episodes": sum(row["losses"] for row in per_run),
                "recovered_episodes": sum(row["recovered"] for row in per_run),
                "maps_created": sum(row["maps_created"] for row in per_run),
                "maps_merged": sum(row["maps_merged"] for row in per_run),
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
