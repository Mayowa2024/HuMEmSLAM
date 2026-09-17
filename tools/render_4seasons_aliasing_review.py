#!/usr/bin/env python3
"""Render 4Seasons car-park footage with ground-truth cross-floor warnings."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--orb-log", type=Path, required=True)
    parser.add_argument("--aliasing-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--xy-threshold-m", type=float, default=2.0)
    parser.add_argument("--floor-height-m", type=float, default=2.4)
    parser.add_argument("--min-frame-separation", type=int, default=300)
    parser.add_argument(
        "--event-hold-frames", type=int, default=30,
        help="Show an event from its logged frame onward for this many frames.",
    )
    return parser.parse_args()


def camera_times(path):
    return np.asarray([
        float(line.split()[1]) for line in path.read_text().splitlines()
        if line.strip()
    ])


def positions(path, times):
    reference = np.loadtxt(path)
    return np.column_stack([
        np.interp(times, reference[:, 0], reference[:, axis])
        for axis in range(1, 4)
    ])


def log_events(path):
    accepted = {}
    rejected = []
    last_sent = None
    for line in path.read_text(errors="replace").splitlines():
        sent = re.search(r"Sending\s+(\d+)\s+stereo pairs", line)
        if sent:
            last_sent = int(sent.group(1))
        match = re.search(
            r"Loop closure metrics:.*current_frame=(\d+);.*matched_kf=(\d+);"
            r".*matched_map_points=(\d+)", line)
        if match:
            accepted[int(match.group(1))] = {
                "matched_kf": int(match.group(2)),
                "map_points": int(match.group(3)),
            }
        elif "BAD LOOP!!!" in line and last_sent is not None:
            rejected.append(last_sent)
    return accepted, rejected


def exact_events(path):
    accepted, rejected = {}, []
    if not path or not path.is_file():
        return accepted, rejected
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                frame = int(row["query_frame_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("accepted") == "1":
                accepted[frame] = {
                    "matched_kf": row.get("candidate_keyframe_id", "?"),
                    "map_points": "",
                    "candidate_frame": row.get("candidate_frame_id", "?"),
                    "cross_floor": row.get("cross_floor_alias") == "1",
                }
            else:
                rejected.append(frame)
    return accepted, rejected


def previous_cross_floor_matches(xyz, xy_threshold, floor_height, separation):
    matches = [None] * len(xyz)
    cell = xy_threshold
    buckets = {}
    for frame, point in enumerate(xyz):
        key = tuple(np.floor(point[:2] / cell).astype(int))
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for prior in buckets.get((key[0] + dx, key[1] + dy), []):
                    if frame - prior < separation:
                        continue
                    dxy = float(np.linalg.norm(point[:2] - xyz[prior, :2]))
                    dz = float(abs(point[2] - xyz[prior, 2]))
                    if dxy <= xy_threshold and dz >= floor_height:
                        score = (dxy, -dz)
                        if best is None or score < best[0]:
                            best = (score, prior, dxy, dz)
        if best:
            matches[frame] = best[1:]
        buckets.setdefault(key, []).append(frame)
    return matches


def text(image, value, y, colour=(235, 235, 235), scale=.58, thickness=1):
    cv2.putText(image, value, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thickness, cv2.LINE_AA)


def main():
    args = arguments()
    images = sorted((args.sequence / "undistorted_images/cam0").glob("*.png"),
                    key=lambda path: int(path.stem))
    times = camera_times(args.sequence / "times.txt")
    count = min(len(images), len(times))
    images, times = images[:count], times[:count]
    xyz = positions(args.sequence / "result.txt", times)
    matches = previous_cross_floor_matches(
        xyz, args.xy_threshold_m, args.floor_height_m,
        args.min_frame_separation)
    accepted, rejected = exact_events(args.aliasing_events)
    exact_decisions = bool(accepted or rejected)
    if not exact_decisions:
        accepted, rejected = log_events(args.orb_log)

    sample = cv2.imread(str(images[0]))
    height, width = sample.shape[:2]
    panel_height = 195
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height + panel_height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create {args.output}")

    rejected_set = set(rejected)
    for frame, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        text(panel, f"4Seasons aliasing review | frame {frame} | elevation {xyz[frame,2]:.2f} m", 25)
        match = matches[frame]
        if match:
            prior, dxy, dz = match
            colour = (0, 165, 255)
            cv2.rectangle(image, (2, 2), (width - 3, height - 3), colour, 4)
            text(panel, "CROSS-FLOOR PROXIMITY (not proof of a proposed match)",
                 56, colour, .58, 2)
            text(panel, f"Earlier frame {prior}: XY separation {dxy:.2f} m, elevation difference {dz:.2f} m", 84, colour)
            inset = cv2.imread(str(images[prior]))
            if inset is not None:
                inset_w, inset_h = min(260, width // 3), min(130, height // 3)
                inset = cv2.resize(inset, (inset_w, inset_h))
                image[8:8 + inset_h, width - inset_w - 8:width - 8] = inset
                cv2.rectangle(image, (width - inset_w - 8, 8),
                              (width - 8, 8 + inset_h), colour, 2)
                cv2.putText(image, f"earlier frame {prior}",
                            (width - inset_w - 3, 8 + inset_h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
        else:
            text(panel, "No close earlier XY position on a different floor", 56, (160, 190, 160))

        near_rejected = [value for value in rejected_set
                         if value <= frame < value + args.event_hold_frames]
        near_accepted = [(value, data) for value, data in accepted.items()
                         if value <= frame < value + args.event_hold_frames]
        if near_rejected:
            marker = max(near_rejected)
            cv2.rectangle(image, (2, 2), (width - 3, height - 3),
                          (40, 40, 255), 5)
            denied_label = (f"ORB LOOP DENIED at exact query frame {marker}"
                            if exact_decisions else
                            f"POSSIBLE LOOP DENIED near frame {marker}")
            text(panel, denied_label,
                 116, (60, 80, 255), .56, 2)
            if exact_decisions:
                text(panel, "Query/candidate identity and floor metrics saved in aliasing_evaluation/events.csv",
                     140, (60, 80, 255), .43, 1)
            else:
                text(panel, "Exact query/candidate frames unavailable in this older log",
                     140, (60, 80, 255), .48, 1)
        if near_accepted:
            marker, data = max(near_accepted, key=lambda item: item[0])
            cv2.rectangle(image, (7, 7), (width - 8, height - 8),
                          (80, 255, 100), 3)
            event_y = 116 if not near_rejected else 170
            candidate = data.get("candidate_frame", "?")
            text(panel, f"ORB native loop ACCEPTED at frame {marker}; candidate frame {candidate}, KF {data['matched_kf']}",
                 event_y, (80, 255, 100), .52, 2)
        if not near_rejected and not near_accepted:
            text(panel, "Green=accepted loop; red=denied loop interval; orange=GT cross-floor proximity",
                 144, (190, 190, 190), .50)
        writer.write(np.vstack((image, panel)))
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
