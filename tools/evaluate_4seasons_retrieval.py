#!/usr/bin/env python3
"""Evaluate HuMemSLAM candidate ranking against 4Seasons ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path,
                        help="HuMemSLAM human_candidates.csv")
    source.add_argument("--events", type=Path,
                        help="ORB-SLAM3 orb_events.csv native-BoW retrieval log")
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--frame-association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-threshold-m", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--min-frame-separation", type=int, default=100)
    return parser.parse_args()


def camera_times(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields:
            values.append(float(fields[1]) if len(fields) > 1 else float(fields[0]))
    return np.asarray(values, dtype=np.float64)


def frame_associations(path: Path) -> dict[int, int]:
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            match = re.search(r"dataset_frame=(\d+)", row.get("source_frame_id", ""))
            if match:
                result[int(row["frame_id"])] = int(match.group(1))
    return result


def reference_poses(sequence: Path) -> tuple[np.ndarray, np.ndarray]:
    times = camera_times(sequence / "times.txt")
    reference = np.loadtxt(sequence / "result.txt")
    indices = np.searchsorted(reference[:, 0], times)
    indices = np.clip(indices, 1, len(reference) - 1)
    before = indices - 1
    choose_before = abs(reference[before, 0] - times) <= abs(reference[indices, 0] - times)
    indices[choose_before] = before[choose_before]
    return reference[indices, 1:4], reference[indices, 4:8]


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def main():
    args = arguments()
    positions, quaternions = reference_poses(args.sequence)
    association = frame_associations(args.frame_association)

    historical_frames = []
    with args.keyframes.open(newline="") as stream:
        for row in csv.DictReader(stream):
            internal = int(row["frame_id"])
            if internal in association:
                historical_frames.append(association[internal])
    historical_frames = sorted(set(historical_frames))
    historical_frames = [frame for frame in historical_frames
                         if 0 <= frame < len(positions)]
    historical_array = np.asarray(historical_frames, dtype=np.int64)
    historical_tree = (
        cKDTree(positions[historical_array]) if len(historical_array) else None
    )

    queries = defaultdict(list)
    if args.candidates:
        with args.candidates.open(newline="") as stream:
            for row in csv.DictReader(stream):
                query = int(row["query_frame_id"])
                candidate = row.get("candidate_source_frame_id", "")
                if candidate:
                    queries[query].append((int(row["rank"]), int(candidate)))
    else:
        with args.events.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("event") not in ("RETRIEVAL_CANDIDATE", "RETRIEVAL_EMPTY"):
                    continue
                details = dict(re.findall(
                    r"([A-Za-z0-9_]+)=([^;]+)", row.get("details", "")))
                if details.get("source") != "native_bow":
                    continue
                query_internal = int(details.get("query_frame", row["frame_id"]))
                query = association.get(query_internal, query_internal)
                queries.setdefault(query, [])
                if row.get("event") == "RETRIEVAL_CANDIDATE":
                    candidate_internal = int(details["candidate_frame"])
                    candidate = association.get(candidate_internal, candidate_internal)
                    queries[query].append((int(details["rank"]), candidate))

    def same_place(query: int, candidate: int) -> bool:
        if min(query, candidate) < 0 or max(query, candidate) >= len(positions):
            return False
        if query - candidate < args.min_frame_separation:
            return False
        distance = float(np.linalg.norm(positions[query] - positions[candidate]))
        angle = quaternion_angle_deg(quaternions[query], quaternions[candidate])
        return distance <= args.position_threshold_m and angle <= args.rotation_threshold_deg

    eligible = []
    labels = {}
    first_ranks = []
    for query, ranked_rows in sorted(queries.items()):
        if not 0 <= query < len(positions) or historical_tree is None:
            continue
        nearby_indices = historical_tree.query_ball_point(
            positions[query], args.position_threshold_m)
        nearby_frames = (
            int(historical_array[index]) for index in nearby_indices)
        has_available_loop = any(
            query - frame >= args.min_frame_separation
            and quaternion_angle_deg(quaternions[query], quaternions[frame])
            <= args.rotation_threshold_deg
            for frame in nearby_frames
        )
        if not has_available_loop:
            continue
        eligible.append(query)
        ranked = [candidate for _, candidate in sorted(ranked_rows)]
        labels[query] = [same_place(query, candidate) for candidate in ranked]
        if any(labels[query]):
            first_ranks.append(labels[query].index(True) + 1)

    metrics = {
        "label_definition": {
            "position_threshold_m": args.position_threshold_m,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "min_frame_separation": args.min_frame_separation,
            "ground_truth": "nearest timestamped 4Seasons reference pose",
        },
        "queries_logged": len(queries),
        "eligible_revisit_queries": len(eligible),
        "frame_association_entries": len(association),
        "keyframe_source_frames": len(historical_frames),
        "retrieval_source": "humanslam" if args.candidates else "native_bow",
        "mrr": (sum(1.0 / rank for rank in first_ranks) / len(eligible)
                if eligible else None),
        "median_first_correct_rank": (
            float(np.median(first_ranks)) if first_ranks else None),
    }
    for k in (1, 3, 5):
        metrics[f"recall@{k}"] = (
            sum(any(labels[query][:k]) for query in eligible) / len(eligible)
            if eligible else None)
        returned = [value for query in eligible for value in labels[query][:k]]
        metrics[f"precision@{k}"] = (
            sum(returned) / len(returned) if returned else None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
