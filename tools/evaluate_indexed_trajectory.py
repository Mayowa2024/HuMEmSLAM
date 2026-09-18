#!/usr/bin/env python3
"""Evaluate KITTI poses using their exported frame/timestamp/map index."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


METRIC = re.compile(r"^\s*(max|mean|median|min|rmse|sse|std)\s+([-+0-9.eE]+)\s*$")
DATASET_FRAME = re.compile(r"(?:^|\|)dataset_frame=(\d+)(?:$|\|)")


def poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    if data.size == 0:
        return np.empty((0, 12), dtype=float)
    return np.atleast_2d(data).reshape(-1, 12)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--times", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-4)
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--expected-end", type=int, default=-1)
    return parser.parse_args()


def main():
    args = arguments()
    estimate = poses(args.trajectory)
    ground_truth = poses(args.ground_truth)
    dataset_times = np.loadtxt(args.times, dtype=float).reshape(-1)
    with args.frame_index.open(newline="", encoding="utf-8") as stream:
        index = list(csv.DictReader(stream))
    if len(index) != len(estimate):
        raise SystemExit(
            f"Trajectory/index mismatch: {len(estimate)} poses, {len(index)} index rows")
    args.output.mkdir(parents=True, exist_ok=True)
    matched = []
    association_methods = set()
    for pose, row in zip(estimate, index):
        timestamp = float(row["dataset_time"])
        source_match = DATASET_FRAME.search(row.get("source_frame_id", ""))
        if source_match:
            dataset_id = int(source_match.group(1))
            association_methods.add("source_frame_id")
        else:
            dataset_id = int(np.argmin(np.abs(dataset_times - timestamp)))
            difference = abs(float(dataset_times[dataset_id]) - timestamp)
            if difference > args.timestamp_tolerance:
                raise SystemExit(
                    f"No dataset timestamp match for {timestamp:.9f}; nearest difference={difference}")
            association_methods.add("timestamp")
        if dataset_id >= len(ground_truth):
            raise SystemExit(f"Ground truth missing dataset frame {dataset_id}")
        matched.append((int(row["trajectory_row"]), int(row["frame_id"]),
                        dataset_id, timestamp, int(row["map_id"]), pose))

    dataset_ids = [item[2] for item in matched]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise SystemExit("Trajectory index contains duplicate dataset frame IDs")

    with (args.output / "matched_frame_index.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trajectory_row", "orb_frame_id", "dataset_frame_id",
                         "dataset_time", "map_id"])
        writer.writerows(item[:5] for item in matched)

    maps = sorted({item[4] for item in matched})
    expected_end = (len(dataset_times) - 1 if args.expected_end < 0 else
                    min(args.expected_end, len(dataset_times) - 1))
    expected_count = max(0, expected_end - args.expected_start + 1)
    summary = {
        "trajectory_pose_count": len(matched),
        "ground_truth_pose_count": len(ground_truth),
        "expected_frame_count": expected_count,
        "expected_frame_range": [args.expected_start, expected_end],
        "tracking_completeness": len(matched) / expected_count if expected_count else None,
        "map_count": len(maps),
        "association_methods": sorted(association_methods),
        "global_ape_valid": len(maps) == 1,
        "maps": {},
    }
    for map_id in maps:
        subset = [item for item in matched if item[4] == map_id]
        map_dir = args.output / f"map_{map_id}"
        map_dir.mkdir(exist_ok=True)
        estimated_path = map_dir / "estimate_kitti.txt"
        truth_path = map_dir / "groundtruth_matched_kitti.txt"
        np.savetxt(estimated_path, np.vstack([item[5] for item in subset]), fmt="%.9f")
        np.savetxt(truth_path,
                   np.vstack([ground_truth[item[2]] for item in subset]), fmt="%.9f")
        record = {
            "pose_count": len(subset),
            "first_dataset_frame": min(item[2] for item in subset),
            "last_dataset_frame": max(item[2] for item in subset),
            "ape": None,
        }
        if shutil.which("evo_ape") and len(subset) >= 3:
            result = subprocess.run(
                ["evo_ape", "kitti", str(truth_path), str(estimated_path), "-a"],
                text=True, capture_output=True,
            )
            (map_dir / "ape_metrics.txt").write_text(
                result.stdout + result.stderr, encoding="utf-8")
            metrics = {}
            for line in result.stdout.splitlines():
                match = METRIC.match(line)
                if match:
                    metrics[match.group(1)] = float(match.group(2))
            record["ape"] = metrics if result.returncode == 0 else None
            record["ape_exit_code"] = result.returncode
        summary["maps"][str(map_id)] = record
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
