#!/usr/bin/env python3
"""Find raw ORB retrieval candidates whose GT poses lie on different floors."""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate_4seasons_aliasing import camera_times, fields, source_frame_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--frame-association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-xy-m", type=float, default=8.0)
    parser.add_argument("--min-height-m", type=float, default=2.4)
    args = parser.parse_args()

    reference = np.loadtxt(args.sequence / "result.txt")
    times = camera_times(args.sequence / "times.txt")
    frame_map = source_frame_map(args.frame_association)
    keyframes = {0: 0}
    with args.keyframes.open(newline="") as stream:
        for row in csv.DictReader(stream):
            keyframes[int(row["keyframe_id"])] = int(row["frame_id"])

    def pose_for_kf(keyframe_id):
        internal = keyframes.get(keyframe_id)
        if internal is None:
            return None
        dataset_frame = frame_map.get(internal, internal)
        if not 0 <= dataset_frame < len(times):
            return None
        index = int(np.argmin(abs(reference[:, 0] - times[dataset_frame])))
        return internal, dataset_frame, reference[index, 1:4]

    rows = []
    seen = set()
    with args.events.open(newline="") as stream:
        for event in csv.DictReader(stream):
            if event.get("event") != "RETRIEVAL_CANDIDATE":
                continue
            detail = fields(event.get("details"))
            if detail.get("source") != "native_bow":
                continue
            query_kf = int(detail["query_kf"])
            candidate_kf = int(detail["candidate_kf"])
            pair = (query_kf, candidate_kf)
            if pair in seen:
                continue
            seen.add(pair)
            query = pose_for_kf(query_kf)
            candidate = pose_for_kf(candidate_kf)
            if query is None or candidate is None:
                continue
            xy = float(np.linalg.norm(query[2][:2] - candidate[2][:2]))
            dz = float(abs(query[2][2] - candidate[2][2]))
            if xy > args.max_xy_m or dz < args.min_height_m:
                continue
            rows.append({
                "query_kf": query_kf,
                "candidate_kf": candidate_kf,
                "query_internal_frame": query[0],
                "candidate_internal_frame": candidate[0],
                "query_dataset_frame": query[1],
                "candidate_dataset_frame": candidate[1],
                "retrieval_score": float(detail.get("retrieval_score", 0.0)),
                "xy_distance_m": xy,
                "height_difference_m": dz,
                "query_z_m": float(query[2][2]),
                "candidate_z_m": float(candidate[2][2]),
            })

    rows.sort(key=lambda row: (-row["retrieval_score"], row["xy_distance_m"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else [
        "query_kf", "candidate_kf", "query_internal_frame",
        "candidate_internal_frame", "query_dataset_frame",
        "candidate_dataset_frame", "retrieval_score", "xy_distance_m",
        "height_difference_m", "query_z_m", "candidate_z_m"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} cross-floor raw candidates -> {args.output}")


if __name__ == "__main__":
    main()
