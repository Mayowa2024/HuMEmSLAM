#!/usr/bin/env python3
"""Export ORB keyframe IDs with processed and original dataset frame IDs."""

import argparse
import csv
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb-events", required=True, type=Path)
    parser.add_argument("--human-latency", required=True, type=Path)
    parser.add_argument("--semantic-frames", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--times", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    times = []
    for line in args.times.read_text().splitlines():
        fields = line.split()
        if fields:
            times.append(float(fields[1] if len(fields) > 1 else fields[0]))

    tracks = {}
    first_wall = None
    with args.orb_events.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["event"] != "track":
                continue
            frame_id = int(row["frame_id"])
            wall = int(row["wall_time_ns"]) * 1e-9
            if first_wall is None:
                first_wall = wall
            tracks[frame_id] = row

    keyframe_frames = set()
    with args.human_latency.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["is_keyframe"]):
                keyframe_frames.add(int(row["query_frame_id"]))

    rows = []
    for processed_id in sorted(keyframe_frames):
        track = tracks.get(processed_id)
        debug_path = args.semantic_frames / f"{processed_id:06d}.png"
        if track is None or not debug_path.exists():
            continue
        wall = int(track["wall_time_ns"]) * 1e-9
        elapsed = wall - first_wall


        best_id = max(0, int(np.searchsorted(times, times[0] + elapsed)) - 1)
        best_id = min(best_id, len(images) - 1)
        rows.append({
            "orb_keyframe_id": track["reference_keyframe_id"],
            "orb_processed_frame_id": processed_id,
            "dataset_frame_id": best_id,
            "dataset_timestamp_s": f"{times[best_id]:.9f}",
            "dataset_image_path": str(images[best_id]),
            "tracking_state": track["tracking_state_name"],
            "map_id": track["map_id"],
            "semantic_debug_path": str(debug_path),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} keyframes to {args.output}")


if __name__ == "__main__":
    main()
