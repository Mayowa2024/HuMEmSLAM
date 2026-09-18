#!/usr/bin/env python3
"""Evaluate tracking-loss and relocalisation episodes for either SLAM mode.

The same evaluator is used for native ORB-SLAM3 and HuMemSLAM runs. A tracking
recovery is not automatically called a correct relocalisation: when a
structured RELOCALIZATION_SUCCESS supplies its matched keyframe, the query and
candidate are independently labelled using dataset ground-truth poses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


LOST_STATES = {"RECENTLY_LOST", "LOST"}
TRACKING_EVENTS = {
    "TRACKING_STATUS", "TRACKING_STABLE", "TRACKING_WEAK", "LOW_INLIERS",
    "TRACKING_RECENTLY_LOST", "TRACKING_LOST",
}


def parse_details(value: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in (value or "").split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed


def integer(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open(encoding="utf-8") as stream:
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


def source_frame(value: str) -> int | None:
    match = re.search(r"(?:dataset_frame=)?(-?\d+)$", value.strip())
    return int(match.group(1)) if match else None


def load_frame_mapping(path: Path | None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not path or not path.is_file():
        return mapping
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = integer(row.get("frame_id"))
            source = source_frame(row.get("source_frame_id", ""))
            if frame >= 0 and source is not None:
                mapping[frame] = source
    return mapping


def load_keyframes(path: Path, frame_mapping: dict[int, int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            keyframe = integer(row.get("keyframe_id"))
            frame = integer(row.get("frame_id"))
            if keyframe >= 0 and frame >= 0:
                mapping[keyframe] = frame_mapping.get(frame, frame)
    return mapping


def gt_pair_label(query: int, candidate: int, poses: list[np.ndarray],
                  position_threshold: float, rotation_threshold: float) -> dict:
    result = {
        "query_source_frame": query,
        "candidate_source_frame": candidate,
        "position_error_m": None,
        "rotation_error_deg": None,
        "ground_truth_correct": None,
    }
    if min(query, candidate) < 0 or max(query, candidate) >= len(poses):
        return result
    position = float(np.linalg.norm(poses[query][:3, 3] - poses[candidate][:3, 3]))
    rotation = rotation_error_deg(poses[query], poses[candidate])
    result.update({
        "position_error_m": position,
        "rotation_error_deg": rotation,
        "ground_truth_correct": (
            position <= position_threshold and rotation <= rotation_threshold
        ),
    })
    return result


def path_distance(start: int, end: int, poses: list[np.ndarray]) -> float | None:
    if start < 0 or end < start or end >= len(poses):
        return None
    return float(sum(
        np.linalg.norm(poses[index][:3, 3] - poses[index - 1][:3, 3])
        for index in range(start + 1, end + 1)
    ))


def evaluate(events_path: Path, keyframes_path: Path, ground_truth_path: Path,
             frame_index_path: Path | None, position_threshold: float,
             rotation_threshold: float) -> dict:
    poses = load_poses(ground_truth_path)
    frame_mapping = load_frame_mapping(frame_index_path)
    keyframes = load_keyframes(keyframes_path, frame_mapping)
    with events_path.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))

    episodes: list[dict] = []
    current: dict | None = None
    relocalisation_events: list[dict] = []
    map_events: list[dict] = []

    for row in events:
        event = row.get("event", "")
        frame = integer(row.get("frame_id"))
        source = frame_mapping.get(frame, frame)
        timestamp = float(row.get("dataset_time") or 0.0)
        details = parse_details(row.get("details"))

        if event in {"MAP_CREATED", "MAP_MERGE_SUCCESS"}:
            map_events.append({"event": event, "frame": source, **details})

        if event.startswith("RELOCALIZATION_"):
            relocal = {
                "event": event,
                "frame": frame,
                "source_frame": source,
                "dataset_time": timestamp,
                **details,
            }
            if event == "RELOCALIZATION_SUCCESS":
                candidate_kf = integer(details.get("candidate_keyframe"))
                candidate_frame = keyframes.get(candidate_kf, -1)
                relocal.update(gt_pair_label(
                    source, candidate_frame, poses,
                    position_threshold, rotation_threshold,
                ))
            relocalisation_events.append(relocal)
            if current is not None:
                current["relocalisation_events"].append(relocal)

        if event not in TRACKING_EVENTS:
            continue
        state = row.get("state_name", "")
        if state in LOST_STATES:
            if current is None:
                current = {
                    "start_frame": frame,
                    "start_source_frame": source,
                    "start_time": timestamp,
                    "initial_state": state,
                    "reached_lost": state == "LOST",
                    "relocalisation_events": [],
                }
            elif state == "LOST":
                current["reached_lost"] = True
        elif state == "OK" and current is not None:
            current.update({
                "recovered": True,
                "end_frame": frame,
                "end_source_frame": source,
                "end_time": timestamp,
                "frames_to_recovery": max(0, source - current["start_source_frame"]),
                "dataset_time_to_recovery_ms": max(
                    0.0, (timestamp - current["start_time"]) * 1000.0),
                "ground_truth_distance_while_lost_m": path_distance(
                    current["start_source_frame"], source, poses),
            })
            successes = [item for item in current["relocalisation_events"]
                         if item["event"] == "RELOCALIZATION_SUCCESS"]
            current["recovery_source"] = (
                successes[-1].get("source", "unknown") if successes
                else "tracking_without_relocalisation"
            )
            current["relocalisation_ground_truth_correct"] = (
                successes[-1].get("ground_truth_correct") if successes else None
            )
            episodes.append(current)
            current = None

    if current is not None:
        current.update({
            "recovered": False,
            "end_frame": None,
            "end_source_frame": None,
            "end_time": None,
            "frames_to_recovery": None,
            "dataset_time_to_recovery_ms": None,
            "ground_truth_distance_while_lost_m": None,
            "recovery_source": "none",
            "relocalisation_ground_truth_correct": None,
        })
        episodes.append(current)

    successes = [item for item in relocalisation_events
                 if item["event"] == "RELOCALIZATION_SUCCESS"]
    correct = [item for item in successes if item.get("ground_truth_correct") is True]
    incorrect = [item for item in successes if item.get("ground_truth_correct") is False]
    unverified = [item for item in successes if item.get("ground_truth_correct") is None]
    recovered = [episode for episode in episodes if episode["recovered"]]
    recovery_frames = [float(item["frames_to_recovery"]) for item in recovered]
    recovery_ms = [float(item["dataset_time_to_recovery_ms"]) for item in recovered]
    recovery_distance = [float(item["ground_truth_distance_while_lost_m"])
                         for item in recovered
                         if item["ground_truth_distance_while_lost_m"] is not None]

    return {
        "label_definition": {
            "position_threshold_m": position_threshold,
            "rotation_threshold_deg": rotation_threshold,
            "note": "Relocalisation correctness compares query and matched historical keyframe ground-truth poses.",
        },
        "tracking_loss": {
            "episodes": len(episodes),
            "episodes_reaching_lost": sum(item["reached_lost"] for item in episodes),
            "recovered_episodes": len(recovered),
            "recovery_rate": len(recovered) / len(episodes) if episodes else None,
            "frames_to_recovery_mean": mean_or_none(recovery_frames),
            "frames_to_recovery_median": median_or_none(recovery_frames),
            "dataset_time_to_recovery_ms_mean": mean_or_none(recovery_ms),
            "dataset_time_to_recovery_ms_median": median_or_none(recovery_ms),
            "distance_while_lost_m_mean": mean_or_none(recovery_distance),
            "recovery_sources": dict(Counter(
                item["recovery_source"] for item in recovered
            )),
        },
        "relocalisation": {
            "attempts": sum(item["event"] == "RELOCALIZATION_ATTEMPT"
                            for item in relocalisation_events),
            "failed_attempts": sum(item["event"] == "RELOCALIZATION_FAILED"
                                   for item in relocalisation_events),
            "successes": len(successes),
            "success_sources": dict(Counter(
                item.get("source", "unknown") for item in successes
            )),
            "ground_truth_correct": len(correct),
            "ground_truth_incorrect": len(incorrect),
            "ground_truth_unverified": len(unverified),
            "correct_success_rate": len(correct) / len(successes) if successes else None,
        },
        "maps": {
            "created": sum(item["event"] == "MAP_CREATED" for item in map_events),
            "merges": sum(item["event"] == "MAP_MERGE_SUCCESS" for item in map_events),
            "merge_sources": dict(Counter(
                item.get("source", "unknown") for item in map_events
                if item["event"] == "MAP_MERGE_SUCCESS"
            )),
            "events": map_events,
        },
        "episodes": episodes,
        "relocalisation_successes": successes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--keyframes", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--frame-index", type=Path)
    parser.add_argument("--position-threshold-m", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.events, args.keyframes, args.ground_truth, args.frame_index,
        args.position_threshold_m, args.rotation_threshold_deg,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
