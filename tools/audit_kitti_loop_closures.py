#!/usr/bin/env python3
"""Audit accepted KITTI loop closures against indexed ground-truth poses."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-distance-m", type=float, default=5.0)
    parser.add_argument("--overlap-distance-m", type=float, default=15.0)
    parser.add_argument("--angle-deg", type=float, default=30.0)
    return parser.parse_args()


def details(value):
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^;]+)", value or ""))


def poses(path):
    raw = np.loadtxt(path).reshape(-1, 3, 4)
    transforms = np.repeat(np.eye(4)[None, :, :], len(raw), axis=0)
    transforms[:, :3, :4] = raw
    return transforms


def frame_map(path):
    mapping = {}
    if not path.exists():
        return mapping
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            match = re.search(r"dataset_frame=(\d+)", row.get("source_frame_id", ""))
            mapping[int(row["frame_id"])] = (
                int(match.group(1)) if match else int(row["frame_id"])
            )
    return mapping


def rotation_angle_deg(first, second):
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def sequence_id(path):
    match = re.search(r"K(\d\d)-", str(path))
    return match.group(1) if match else None


def campaign_id(path):
    for parent in path.parents:
        if re.fullmatch(r"K\d\d-.*", parent.name):
            return parent.name
    return "unknown"


def audit_run(method_dir, gt, args):
    events_path = method_dir / "orb_events.csv"
    keyframes_path = method_dir / "orb_keyframes.csv"
    if not events_path.exists() or not keyframes_path.exists():
        return [], "missing_events_or_keyframes"

    mapping = frame_map(method_dir / "trajectory_frame_ids.csv")
    keyframes = {}
    with keyframes_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            internal = int(row["frame_id"])
            keyframes[int(row["keyframe_id"])] = mapping.get(internal, internal)

    detected = []
    corrected = {}
    with events_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            data = details(row.get("details", ""))
            if row.get("event") == "LOOP_DETECTED":
                detected.append((row, data))
            elif row.get("event") == "LOOP_CORRECTION_METRICS":
                corrected[(data.get("current_kf"), data.get("matched_kf"))] = data

    records = []
    for row, data in detected:
        q_kf = int(data.get("current_kf", -1))
        c_kf = int(data.get("matched_kf", -1))
        correction = corrected.get((str(q_kf), str(c_kf)))
        if correction is None:
            continue
        q_frame = keyframes.get(q_kf)
        c_frame = keyframes.get(c_kf)


        if c_frame is None and correction.get("matched_frame") is not None:
            internal = int(correction["matched_frame"])
            c_frame = mapping.get(internal, internal)
        if q_frame is None:
            internal = int(row.get("frame_id", -1))
            q_frame = mapping.get(internal, internal)
        record = {
            "campaign": campaign_id(method_dir),
            "run": method_dir.parent.name,
            "method": method_dir.name,
            "source": correction.get("source", "unknown"),
            "query_keyframe_id": q_kf,
            "candidate_keyframe_id": c_kf,
            "query_frame_id": q_frame if q_frame is not None else "",
            "candidate_frame_id": c_frame if c_frame is not None else "",
            "distance_3d_m": "", "orientation_difference_deg": "",
            "strict_revisit": "", "valid_place_overlap": "",
            "classification": "unverified",
        }
        if (q_frame is not None and c_frame is not None
                and 0 <= q_frame < len(gt) and 0 <= c_frame < len(gt)):
            distance = float(np.linalg.norm(
                gt[q_frame, :3, 3] - gt[c_frame, :3, 3]
            ))
            angle = rotation_angle_deg(gt[q_frame], gt[c_frame])
            strict = distance <= args.strict_distance_m and angle <= args.angle_deg
            overlap = distance <= args.overlap_distance_m and angle <= args.angle_deg
            record.update({
                "distance_3d_m": distance,
                "orientation_difference_deg": angle,
                "strict_revisit": int(strict),
                "valid_place_overlap": int(overlap),
                "classification": "valid_overlap" if overlap else "false_closure",
            })
        records.append(record)
    return records, "ok"


def main():
    args = arguments()
    records = []
    skipped = []
    audited = set()
    for method_dir in sorted(args.campaign_root.glob("**/run_*/baseline")) + sorted(
        args.campaign_root.glob("**/run_*/humanslam")
    ):
        sequence = sequence_id(method_dir)
        gt_path = args.poses_root / f"{sequence}.txt" if sequence else None
        if gt_path is None or not gt_path.exists():
            skipped.append({"path": str(method_dir), "reason": "missing_ground_truth"})
            continue
        rows, status = audit_run(method_dir, poses(gt_path), args)
        records.extend(rows)
        audited.add((campaign_id(method_dir), method_dir.name))
        if status != "ok":
            skipped.append({"path": str(method_dir), "reason": status})

    args.output.mkdir(parents=True, exist_ok=True)
    columns = [
        "campaign", "run", "method", "source", "query_keyframe_id",
        "candidate_keyframe_id", "query_frame_id", "candidate_frame_id",
        "distance_3d_m", "orientation_difference_deg", "strict_revisit",
        "valid_place_overlap", "classification",
    ]
    with (args.output / "accepted_loops.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)

    aggregate = defaultdict(lambda: {"accepted": 0, "valid_overlap": 0,
                                     "false_closure": 0, "unverified": 0})
    for key in audited:
        aggregate[key]
    for row in records:
        key = (row["campaign"], row["method"])
        aggregate[key]["accepted"] += 1
        aggregate[key][row["classification"]] += 1
    summary_rows = [
        {"campaign": key[0], "method": key[1], **values}
        for key, values in sorted(aggregate.items())
    ]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        columns = ["campaign", "method", "accepted", "valid_overlap",
                   "false_closure", "unverified"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(summary_rows)
    (args.output / "summary.json").write_text(json.dumps({
        "thresholds": {"strict_distance_m": args.strict_distance_m,
                       "overlap_distance_m": args.overlap_distance_m,
                       "angle_deg": args.angle_deg},
        "accepted_events": len(records), "summary": summary_rows,
        "skipped": skipped,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
