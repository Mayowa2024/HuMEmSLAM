#!/usr/bin/env python3
"""Evaluate loop-candidate retrieval separately from trajectory accuracy.

The evaluator labels a candidate as a true revisit from ground-truth poses,
then reports ranking, geometric-verification, closure, correction, and latency
metrics from ORB-SLAM3's orb_events.csv instrumentation.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_details(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in (value or "").split(";"):
        if "=" in item:
            key, val = item.strip().split("=", 1)
            result[key.strip()] = val.strip()
    return result


def load_poses(path: Path) -> list[np.ndarray]:
    poses = []
    with path.open() as stream:
        for line in stream:
            values = [float(value) for value in line.split()]
            if len(values) != 12:
                raise ValueError(f"Expected KITTI 3x4 pose, got {len(values)} values")
            pose = np.eye(4)
            pose[:3, :] = np.asarray(values).reshape(3, 4)
            poses.append(pose)
    return poses


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True,
                        help="KITTI poses.txt (one 3x4 camera pose per frame)")
    parser.add_argument(
        "--times", type=Path,
        help="Dataset timestamps used to map ORB internal frames to source frames.",
    )
    parser.add_argument("--position-threshold-m", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--min-frame-separation", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    poses = load_poses(args.ground_truth)
    with args.keyframes.open(newline="") as stream:
        keyframes = list(csv.DictReader(stream))
    timestamps = None
    if args.times:
        timestamps = [
            float(line) for line in args.times.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def source_frame(dataset_time: str | float, fallback: str | int) -> int:
        if not timestamps:
            return int(fallback)
        value = float(dataset_time)
        index = bisect.bisect_left(timestamps, value)
        choices = [candidate for candidate in (index - 1, index)
                   if 0 <= candidate < len(timestamps)]
        return min(choices, key=lambda candidate: abs(timestamps[candidate] - value))

    keyframe_sources = {
        int(row["keyframe_id"]): source_frame(row["dataset_time"], row["frame_id"])
        for row in keyframes
    }
    keyframe_frames = sorted(set(keyframe_sources.values()))

    events = []
    with args.events.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["parsed"] = parse_details(row.get("details", ""))
            events.append(row)

    candidates: dict[tuple[str, int], list[dict]] = defaultdict(list)
    geometry = []
    corrections = []
    for event in events:
        details = event["parsed"]
        if event["event"] == "RETRIEVAL_CANDIDATE":
            source = details.get("source", "unknown")
            query = source_frame(
                event.get("dataset_time", ""),
                details.get("query_frame", event["frame_id"]),
            )
            candidate_kf = details.get("candidate_kf")
            if candidate_kf is not None and int(candidate_kf) in keyframe_sources:
                details["candidate_source_frame"] = str(
                    keyframe_sources[int(candidate_kf)]
                )
            else:
                details["candidate_source_frame"] = details.get(
                    "candidate_frame", "-1"
                )
            candidates[(source, query)].append(details)
        elif event["event"] == "RETRIEVAL_EMPTY":
            source = details.get("source", "unknown")
            query = source_frame(
                event.get("dataset_time", ""),
                details.get("query_frame", event["frame_id"]),
            )
            candidates.setdefault((source, query), [])
        elif event["event"] == "RETRIEVAL_GEOMETRIC_RESULT":
            geometry.append(details)
        elif event["event"] == "LOOP_CORRECTION_METRICS":
            corrections.append(details)

    def is_positive(query: int, candidate: int) -> bool:
        if min(query, candidate) < 0 or max(query, candidate) >= len(poses):
            return False
        if query - candidate < args.min_frame_separation:
            return False
        distance = float(np.linalg.norm(poses[query][:3, 3] - poses[candidate][:3, 3]))
        angle = rotation_error_deg(poses[query], poses[candidate])
        return distance <= args.position_threshold_m and angle <= args.rotation_threshold_deg

    sources = sorted({source for source, _ in candidates})
    output: dict[str, object] = {
        "label_definition": {
            "position_threshold_m": args.position_threshold_m,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "min_frame_separation": args.min_frame_separation,
        },
        "sources": {},
    }

    for source in sources:
        source_queries = sorted(query for src, query in candidates if src == source)
        eligible = []
        labelled: dict[int, list[bool]] = {}
        ranks = []
        latencies = []
        for query in source_queries:
            prior = [frame for frame in keyframe_frames
                     if query - frame >= args.min_frame_separation]
            has_available_loop = any(is_positive(query, frame) for frame in prior)
            ranked = sorted(candidates[(source, query)],
                            key=lambda item: int(item.get("rank", 10**9)))
            labels = [
                is_positive(query, int(item["candidate_source_frame"]))
                for item in ranked
            ]
            labelled[query] = labels
            if has_available_loop:
                eligible.append(query)
                if any(labels):
                    ranks.append(labels.index(True) + 1)
            retrieval_ms = [float(item["retrieval_ms"]) for item in ranked
                            if "retrieval_ms" in item]
            if retrieval_ms:
                latencies.append(retrieval_ms[0])

        metrics: dict[str, object] = {
            "queries_logged": len(source_queries),
            "eligible_revisit_queries": len(eligible),
            "proposal_coverage": (sum(bool(labelled[q]) for q in eligible) / len(eligible)
                                  if eligible else None),
            "mrr": (sum(1.0 / rank for rank in ranks) / len(eligible)
                    if eligible else None),
            "median_first_correct_rank": median_or_none([float(rank) for rank in ranks]),
            "retrieval_latency_ms_mean": mean_or_none(latencies),
            "retrieval_latency_ms_median": median_or_none(latencies),
        }
        for k in (1, 3, 5):
            metrics[f"recall@{k}"] = (
                sum(any(labelled[q][:k]) for q in eligible) / len(eligible)
                if eligible else None)
            returned = [label for q in eligible for label in labelled[q][:k]]
            metrics[f"precision@{k}"] = (sum(returned) / len(returned)
                                           if returned else None)
        output["sources"][source] = metrics

    geometry_ms = [float(item["geometry_ms"]) for item in geometry
                   if "geometry_ms" in item]
    verified = [item for item in geometry
                if int(item.get("temporal_consistency", "0")) > 0]
    accepted = [item for item in geometry if item.get("accepted_seed") == "1"]
    output["geometric_verification"] = {
        "attempts": len(geometry),
        "verified_attempts": len(verified),
        "accepted_seeds": len(accepted),
        "verification_rate": len(verified) / len(geometry) if geometry else None,
        "seed_acceptance_rate": len(accepted) / len(geometry) if geometry else None,
        "latency_ms_mean": mean_or_none(geometry_ms),
        "latency_ms_median": median_or_none(geometry_ms),
    }

    translation = [float(item["applied_translation_m"]) for item in corrections
                   if "applied_translation_m" in item]
    rotation = [math.degrees(float(item["applied_rotation_rad"])) for item in corrections
                if "applied_rotation_rad" in item]
    correction_ms = [float(item["correction_ms"]) for item in corrections
                     if "correction_ms" in item]
    output["loop_correction"] = {
        "closures": len(corrections),
        "translation_m_mean": mean_or_none(translation),
        "rotation_deg_mean": mean_or_none(rotation),
        "latency_ms_mean": mean_or_none(correction_ms),
        "latency_ms_median": median_or_none(correction_ms),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
