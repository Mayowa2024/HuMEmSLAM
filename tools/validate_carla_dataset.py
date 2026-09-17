#!/usr/bin/env python3
"""Fail-fast integrity check for a recorded HuMemSLAM CARLA dataset."""

import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()

    required = ["image_0", "image_1", "times.txt", "pose_gt.csv", "imu.csv",
                "frame_metadata.csv", "lap_events.csv", "metadata.json"]
    missing = [name for name in required if not (root / name).exists()]
    errors = [f"missing: {', '.join(missing)}"] if missing else []
    if missing:
        raise SystemExit("INVALID\n" + "\n".join(errors))

    left = sorted((root / "image_0").glob("*.png"))
    right = sorted((root / "image_1").glob("*.png"))
    times = [line for line in (root / "times.txt").read_text().splitlines()
             if line.strip()]
    poses = rows(root / "pose_gt.csv")
    frames = rows(root / "frame_metadata.csv")
    events = rows(root / "lap_events.csv")
    metadata = json.loads((root / "metadata.json").read_text())

    if [p.name for p in left] != [p.name for p in right]:
        errors.append("left/right image filenames differ")
    counts = {"left": len(left), "right": len(right), "times": len(times),
              "poses": len(poses), "frame_metadata": len(frames)}
    if len(set(counts.values())) != 1:
        errors.append(f"stereo/metadata counts differ: {counts}")
    expected = [str(index) for index in range(len(left))]
    for label, data in (("pose", poses), ("frame metadata", frames)):
        actual = [row.get("image_index") for row in data]
        if actual != expected:
            errors.append(f"{label} image_index is not contiguous")
    completed = [row for row in events if row.get("event") == "LAP_COMPLETE"]



    weather_order = metadata.get(
        "weather_order", metadata.get("weather_sequence", [])
    )
    if len(completed) != len(weather_order):
        errors.append(
            f"completed laps={len(completed)}, expected={len(weather_order)}")
    if not args.allow_incomplete:
        if (root / "RECORDING_IN_PROGRESS").exists():
            errors.append("RECORDING_IN_PROGRESS marker remains")
        if not (root / "DATASET_COMPLETE").exists():
            errors.append("DATASET_COMPLETE marker is absent")
        dataset_status = metadata.get("dataset_status", metadata.get("status"))
        if dataset_status != "COMPLETE":
            errors.append("metadata status is not COMPLETE")

    print(f"dataset={root}")
    print("counts=" + json.dumps(counts, sort_keys=True))
    print(f"completed_laps={len(completed)} weather_order={weather_order}")
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()
