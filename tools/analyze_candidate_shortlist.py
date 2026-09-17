#!/usr/bin/env python3
"""Replay HuMemSLAM CSV ranking with smaller scene-prefilter shortlists."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_poses(path: Path) -> list[np.ndarray]:
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.asarray([float(value) for value in line.split()])
        pose = np.eye(4)
        pose[:3, :] = values.reshape(3, 4)
        poses.append(pose)
    return poses


def correct(poses, query, candidate, position_m, rotation_deg):
    if min(query, candidate) < 0 or max(query, candidate) >= len(poses):
        return False
    query_pose, candidate_pose = poses[query], poses[candidate]
    distance = np.linalg.norm(query_pose[:3, 3] - candidate_pose[:3, 3])
    relative = query_pose[:3, :3].T @ candidate_pose[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.degrees(math.acos(float(cosine)))
    return distance <= position_m and angle <= rotation_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--shortlists", default="5,10,15,25")
    parser.add_argument("--position-threshold-m", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    poses = load_poses(args.ground_truth)
    grouped = defaultdict(list)
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[int(row["query_frame_id"])].append(row)



    cohort = {
        query: rows for query, rows in grouped.items()
        if any(row["published"] == "1" for row in rows)
    }
    output = {"query_cohort": len(cohort), "shortlists": {}}
    for size in [int(value) for value in args.shortlists.split(",")]:
        labels_by_query = []
        submitted_counts = []
        for query, rows in cohort.items():
            scene_prefilter = sorted(
                rows, key=lambda row: float(row["scene_score"] or 0.0), reverse=True
            )[:size]
            reranked = sorted(
                scene_prefilter,
                key=lambda row: float(row["effective_score"] or 0.0),
                reverse=True,
            )
            submitted = [
                row for row in reranked
                if float(row["effective_score"] or 0.0)
                > float(row["submission_threshold"] or 0.70)
            ][:5]
            submitted_counts.append(len(submitted))
            labels_by_query.append([
                correct(
                    poses, query, int(row["candidate_source_frame_id"]),
                    args.position_threshold_m, args.rotation_threshold_deg,
                )
                for row in submitted
            ])
        count = len(labels_by_query)
        ranks = [
            labels.index(True) + 1 for labels in labels_by_query if any(labels)
        ]
        metrics = {
            "queries": count,
            "proposal_coverage": sum(bool(labels) for labels in labels_by_query) / count,
            "mean_submitted": float(np.mean(submitted_counts)),
            "mrr": sum(1.0 / rank for rank in ranks) / count,
        }
        for k in (1, 3, 5):
            metrics[f"recall@{k}"] = (
                sum(any(labels[:k]) for labels in labels_by_query) / count
            )
        output["shortlists"][str(size)] = metrics

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
