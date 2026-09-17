#!/usr/bin/env python3
"""Compare native BoW and HuMemSLAM on exactly the same 4Seasons queries.

Unlike the independent retrieval evaluator, this tool constructs one shared
denominator. A query is included only when:

1. both methods logged retrieval for the exact same dataset frame; and
2. both method-specific keyframe databases contain a valid earlier keyframe
   under the requested ground-truth position/orientation thresholds.

Each method is then evaluated using its own ranked candidates. This preserves
the end-to-end effect of database construction while removing query-schedule
confounding.
"""

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
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--baseline-keyframes", type=Path, required=True)
    parser.add_argument("--baseline-frame-association", type=Path, required=True)
    parser.add_argument("--human-candidates", type=Path, required=True)
    parser.add_argument("--human-keyframes", type=Path, required=True)
    parser.add_argument("--human-frame-association", type=Path, required=True)
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


def reference_poses(sequence: Path) -> tuple[np.ndarray, np.ndarray]:
    times = camera_times(sequence / "times.txt")
    reference = np.loadtxt(sequence / "result.txt")
    indices = np.searchsorted(reference[:, 0], times)
    indices = np.clip(indices, 1, len(reference) - 1)
    before = indices - 1
    choose_before = abs(reference[before, 0] - times) <= abs(
        reference[indices, 0] - times
    )
    indices[choose_before] = before[choose_before]
    return reference[indices, 1:4], reference[indices, 4:8]


def frame_associations(path: Path) -> dict[int, int]:
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            match = re.search(r"dataset_frame=(\d+)", row.get("source_frame_id", ""))
            if match:
                result[int(row["frame_id"])] = int(match.group(1))
    return result


def keyframe_sources(path: Path, association: dict[int, int]) -> list[int]:
    result = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            internal = int(row["frame_id"])
            if internal in association:
                result.append(association[internal])
    return sorted(set(result))


def native_queries(path: Path, association: dict[int, int]):
    queries = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("event") not in ("RETRIEVAL_CANDIDATE", "RETRIEVAL_EMPTY"):
                continue
            details = dict(re.findall(
                r"([A-Za-z0-9_]+)=([^;]+)", row.get("details", "")
            ))
            if details.get("source") != "native_bow":
                continue
            internal = int(details.get("query_frame", row["frame_id"]))
            query = association.get(internal, internal)
            queries.setdefault(query, [])
            if row.get("event") == "RETRIEVAL_CANDIDATE":
                candidate_internal = int(details["candidate_frame"])
                candidate = association.get(candidate_internal, candidate_internal)
                queries[query].append((int(details["rank"]), candidate))
    return queries


def human_queries(path: Path):
    queries = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            query = int(row["query_frame_id"])
            queries.setdefault(query, [])
            candidate = row.get("candidate_source_frame_id", "")
            if candidate:
                queries[query].append((int(row["rank"]), int(candidate)))
    return queries


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def main():
    args = arguments()
    positions, quaternions = reference_poses(args.sequence)
    baseline_association = frame_associations(args.baseline_frame_association)
    human_association = frame_associations(args.human_frame_association)
    baseline_db = keyframe_sources(args.baseline_keyframes, baseline_association)
    human_db = keyframe_sources(args.human_keyframes, human_association)
    baseline = native_queries(args.baseline_events, baseline_association)
    human = human_queries(args.human_candidates)

    database_trees = {}
    for name, frames in (("baseline", baseline_db), ("humanslam", human_db)):
        valid = np.asarray(
            [frame for frame in frames if 0 <= frame < len(positions)],
            dtype=np.int64,
        )
        database_trees[name] = (
            valid,
            cKDTree(positions[valid]) if len(valid) else None,
        )

    def same_place(query: int, candidate: int) -> bool:
        if min(query, candidate) < 0 or max(query, candidate) >= len(positions):
            return False
        if query - candidate < args.min_frame_separation:
            return False
        distance = float(np.linalg.norm(positions[query] - positions[candidate]))
        angle = quaternion_angle_deg(quaternions[query], quaternions[candidate])
        return (
            distance <= args.position_threshold_m
            and angle <= args.rotation_threshold_deg
        )

    def database_has_loop(name: str, query: int) -> bool:
        frames, tree = database_trees[name]
        if tree is None:
            return False
        for index in tree.query_ball_point(
            positions[query], args.position_threshold_m
        ):
            frame = int(frames[index])
            if (
                query - frame >= args.min_frame_separation
                and quaternion_angle_deg(quaternions[query], quaternions[frame])
                <= args.rotation_threshold_deg
            ):
                return True
        return False

    exact_shared = sorted(set(baseline) & set(human))
    eligible = [
        query for query in exact_shared
        if 0 <= query < len(positions)
        and database_has_loop("baseline", query)
        and database_has_loop("humanslam", query)
    ]

    labels = {"baseline": {}, "humanslam": {}}
    for name, source in (("baseline", baseline), ("humanslam", human)):
        for query in eligible:
            ranked = [candidate for _, candidate in sorted(source[query])]
            labels[name][query] = [same_place(query, candidate) for candidate in ranked]

    def method_metrics(name: str):
        first_ranks = []
        for query in eligible:
            values = labels[name][query]
            if any(values):
                first_ranks.append(values.index(True) + 1)
        result = {
            "mrr": (
                sum(1.0 / rank for rank in first_ranks) / len(eligible)
                if eligible else None
            ),
            "median_first_correct_rank": (
                float(np.median(first_ranks)) if first_ranks else None
            ),
        }
        for k in (1, 3, 5):
            result[f"recall@{k}"] = (
                sum(any(labels[name][query][:k]) for query in eligible) / len(eligible)
                if eligible else None
            )
            returned = [
                value for query in eligible for value in labels[name][query][:k]
            ]
            result[f"precision@{k}"] = (
                sum(returned) / len(returned) if returned else None
            )
        return result

    paired = {}
    for k in (1, 3, 5):
        baseline_success = {
            query: any(labels["baseline"][query][:k]) for query in eligible
        }
        human_success = {
            query: any(labels["humanslam"][query][:k]) for query in eligible
        }
        paired[f"at_{k}"] = {
            "both_correct": sum(
                baseline_success[q] and human_success[q] for q in eligible
            ),
            "baseline_only": sum(
                baseline_success[q] and not human_success[q] for q in eligible
            ),
            "humanslam_only": sum(
                human_success[q] and not baseline_success[q] for q in eligible
            ),
            "neither_correct": sum(
                not baseline_success[q] and not human_success[q] for q in eligible
            ),
        }

    output = {
        "label_definition": {
            "position_threshold_m": args.position_threshold_m,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "min_frame_separation": args.min_frame_separation,
            "query_pairing": "exact dataset source-frame intersection",
            "eligibility": "valid earlier keyframe exists in both databases",
        },
        "baseline_queries_logged": len(baseline),
        "humanslam_queries_logged": len(human),
        "exact_shared_queries": len(exact_shared),
        "shared_eligible_queries": len(eligible),
        "baseline": method_metrics("baseline"),
        "humanslam": method_metrics("humanslam"),
        "paired_outcomes": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
