#!/usr/bin/env python3
"""Label 4Seasons loop proposals using query/candidate ground-truth height."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--keyframes", type=Path, required=True)
    p.add_argument(
        "--frame-association", type=Path,
        help=("trajectory_frame_ids.csv mapping ORB's internal frame counter "
              "to the original dataset_frame index"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--same-place-3d-m", type=float, default=5.0)
    p.add_argument("--same-place-angle-deg", type=float, default=30.0)
    p.add_argument("--valid-overlap-3d-m", type=float, default=15.0)
    p.add_argument("--cross-floor-xy-m", type=float, default=5.0)
    p.add_argument("--floor-height-m", type=float, default=2.4)
    return p.parse_args()


def fields(value):
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^;]+)", value or ""))


def camera_times(path):
    """Read KITTI or 4Seasons camera timestamps without losing precision."""
    values = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields:
            values.append(float(fields[1]) if len(fields) >= 2 else float(fields[0]))
    return np.asarray(values, dtype=np.float64)


def source_frame_map(path):
    """Return ORB internal-frame -> original dataset-frame associations."""
    mapping = {}
    if path is None or not path.exists():
        return mapping
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            internal = int(row["frame_id"])
            source = row.get("source_frame_id", "")
            match = re.search(r"dataset_frame=(\d+)", source)
            if match:
                mapping[internal] = int(match.group(1))
    return mapping


def quaternion_angle_deg(q1, q2):
    """Smallest angular difference for xyzw unit quaternions."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    q2 /= max(np.linalg.norm(q2), 1e-12)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main():
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reference = np.loadtxt(a.sequence / "result.txt")
    times = camera_times(a.sequence / "times.txt")
    frame_map = source_frame_map(a.frame_association)

    keyframes = {}
    with a.keyframes.open(newline="") as stream:
        for row in csv.DictReader(stream):
            keyframes[int(row["keyframe_id"])] = {
                "frame_id": int(row["frame_id"]),
                "time": float(row["dataset_time"]),
            }

    detected, corrected = [], {}
    with a.events.open(newline="") as stream:
        for row in csv.DictReader(stream):
            detail = fields(row.get("details"))
            if row.get("event") == "LOOP_DETECTED":
                detected.append((row, detail))
            elif row.get("event") == "LOOP_CORRECTION_METRICS":
                key = (detail.get("current_kf"), detail.get("matched_kf"))
                corrected[key] = detail

    output = []
    for row, detail in detected:
        current_kf = int(detail.get("current_kf", -1))
        matched_kf = int(detail.get("matched_kf", -1))
        query = keyframes.get(current_kf)
        candidate = keyframes.get(matched_kf)



        if candidate is None and matched_kf == 0:
            candidate = {"frame_id": 0, "time": times[0]}
        accepted = (str(current_kf), str(matched_kf)) in corrected
        record = {
            "query_internal_frame_id": query["frame_id"] if query else row.get("frame_id"),
            "candidate_internal_frame_id": candidate["frame_id"] if candidate else "",
            "query_frame_id": "", "candidate_frame_id": "",
            "query_keyframe_id": current_kf,
            "candidate_keyframe_id": matched_kf,
            "accepted": int(accepted),
            "decision": "accepted" if accepted else "denied",
            "query_z_m": "", "candidate_z_m": "", "xy_distance_m": "",
            "z_difference_m": "", "distance_3d_m": "",
            "orientation_difference_deg": "", "position_same_place": "",
            "strict_revisit": "", "valid_place_overlap": "",
            "cross_floor_alias": "", "ground_truth_same_place": "",
        }
        if query and candidate:





            q_internal = query["frame_id"]
            c_internal = candidate["frame_id"]
            q_frame = frame_map.get(q_internal, q_internal)
            c_frame = frame_map.get(c_internal, c_internal)
            record["query_frame_id"] = q_frame
            record["candidate_frame_id"] = c_frame
            q_time = times[q_frame] if 0 <= q_frame < len(times) else query["time"]
            c_time = times[c_frame] if 0 <= c_frame < len(times) else candidate["time"]
            qi = int(np.argmin(abs(reference[:, 0] - q_time)))
            ci = int(np.argmin(abs(reference[:, 0] - c_time)))
            q, c = reference[qi, 1:4], reference[ci, 1:4]
            angle = quaternion_angle_deg(reference[qi, 4:8], reference[ci, 4:8])
            xy = float(np.linalg.norm(q[:2] - c[:2]))
            dz = float(abs(q[2] - c[2]))
            d3 = float(np.linalg.norm(q - c))
            record.update({
                "query_z_m": float(q[2]), "candidate_z_m": float(c[2]),
                "xy_distance_m": xy, "z_difference_m": dz,
                "distance_3d_m": d3,
                "orientation_difference_deg": angle,
                "position_same_place": int(d3 <= a.same_place_3d_m),
                "strict_revisit": int(
                    d3 <= a.same_place_3d_m
                    and angle <= a.same_place_angle_deg
                ),
                "valid_place_overlap": int(
                    d3 <= a.valid_overlap_3d_m
                    and angle <= a.same_place_angle_deg
                ),
                "cross_floor_alias": int(xy <= a.cross_floor_xy_m and dz >= a.floor_height_m),
                "ground_truth_same_place": int(
                    d3 <= a.same_place_3d_m
                    and angle <= a.same_place_angle_deg
                ),
            })
        output.append(record)

    columns = list(output[0]) if output else [
        "query_internal_frame_id", "candidate_internal_frame_id",
        "query_frame_id", "candidate_frame_id", "query_keyframe_id",
        "candidate_keyframe_id", "accepted", "decision", "query_z_m",
        "candidate_z_m", "xy_distance_m", "z_difference_m", "distance_3d_m",
        "orientation_difference_deg", "position_same_place",
        "strict_revisit", "valid_place_overlap",
        "cross_floor_alias", "ground_truth_same_place"]
    with (a.output_dir / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(output)
    summary = {
        "loop_proposals": len(output),
        "accepted": sum(item["accepted"] == 1 for item in output),
        "denied": sum(item["accepted"] == 0 for item in output),
        "cross_floor_proposals": sum(item["cross_floor_alias"] == 1 for item in output),
        "accepted_cross_floor": sum(item["accepted"] == 1 and item["cross_floor_alias"] == 1 for item in output),
        "denied_cross_floor": sum(item["accepted"] == 0 and item["cross_floor_alias"] == 1 for item in output),
        "ground_truth_same_place": sum(item["ground_truth_same_place"] == 1 for item in output),
        "valid_place_overlap": sum(item["valid_place_overlap"] == 1 for item in output),
        "frame_association_entries": len(frame_map),
        "thresholds": {"same_place_3d_m": a.same_place_3d_m,
                       "same_place_angle_deg": a.same_place_angle_deg,
                       "valid_overlap_3d_m": a.valid_overlap_3d_m,
                       "cross_floor_xy_m": a.cross_floor_xy_m,
                       "floor_height_m": a.floor_height_m},
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(a.output_dir)


if __name__ == "__main__":
    main()
