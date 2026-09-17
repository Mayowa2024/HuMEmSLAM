#!/usr/bin/env python3
"""Render a stable-layout baseline or HuMemSLAM diagnostic replay."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PANEL_HEIGHT = 220


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--orb-events", type=Path)
    parser.add_argument("--orb-features", type=Path)
    parser.add_argument("--semantic-frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="ORB-SLAM3 baseline")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    return parser.parse_args()


def event_states(path):
    states = {}
    if not path or not path.is_file():
        return states
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                frame = int(row["frame_id"])
            except (KeyError, TypeError, ValueError):
                continue
            states[frame] = (
                row.get("state_name")
                or
                row.get("tracking_state_name")
                or row.get("tracking_state")
                or "UNKNOWN"
            )
    return states


def tracked_features(path):
    features = {}
    if not path or not path.is_file():
        return features
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                dataset_id = int(row.get("dataset_frame_id", "-1"))
                processed_id = int(row["frame_id"])
                frame = dataset_id if dataset_id >= 0 else processed_id
                features.setdefault(frame, []).append(
                    (float(row["x"]), float(row["y"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return features


def put_line(panel, text, line, colour=(235, 235, 235)):
    cv2.putText(
        panel, text, (12, 27 + 28 * line), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, colour, 1, cv2.LINE_AA,
    )


def main():
    args = arguments()
    all_images = sorted(args.images.glob("*.png"), key=lambda p: int(p.stem))
    final = len(all_images) if args.end_frame < 0 else args.end_frame + 1
    images = all_images[max(0, args.start_frame):final]
    if not images:
        raise SystemExit(f"No PNG images in {args.images}")
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (width, height + PANEL_HEIGHT),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create {args.output}")

    states = event_states(args.orb_events)
    features = tracked_features(args.orb_features)
    current_state = "NO_IMAGES_YET"
    semantic_root = args.semantic_frames
    for frame_id, image_path in enumerate(images, start=max(0, args.start_frame)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if frame_id in states:
            current_state = states[frame_id]
        panel = np.zeros((PANEL_HEIGHT, width, 3), dtype=np.uint8)
        put_line(panel, f"{args.label} | frame={frame_id} | tracking={current_state}", 0)

        semantic_path = (
            semantic_root / f"{frame_id:06d}.png" if semantic_root else None
        )
        if semantic_path and semantic_path.is_file():
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_COLOR)
            if semantic is not None:
                image = semantic[:height, :width]
                footer = semantic[height:, :width]
                copy_height = min(footer.shape[0], PANEL_HEIGHT - 34)
                panel[34:34 + copy_height] = footer[:copy_height]
        else:
            mode = "HuMemSLAM: no semantic query on this frame" if semantic_root else "HuMemSLAM: disabled"
            put_line(panel, mode, 1, (180, 220, 255))



        for x, y in features.get(frame_id, []):
            cv2.circle(image, (int(round(x)), int(round(y))), 2,
                       (60, 255, 80), -1, cv2.LINE_AA)
        put_line(panel, f"Tracked ORB map features: {len(features.get(frame_id, []))}",
                 6, (80, 255, 120))

        writer.write(np.vstack((image, panel)))
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
