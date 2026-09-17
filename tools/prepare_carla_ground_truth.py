#!/usr/bin/env python3
"""Convert recorder pose_gt.csv files to right-handed KITTI camera poses."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np





CARLA_TO_OPTICAL = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float
)


def carla_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Match CARLA Transform.get_matrix() rotation convention."""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=float,
    )


def convert_row(row: dict[str, str]) -> np.ndarray:
    rotation_carla = carla_rotation(
        float(row["roll"]), float(row["pitch"]), float(row["yaw"])
    )
    rotation_optical = (
        CARLA_TO_OPTICAL @ rotation_carla @ CARLA_TO_OPTICAL.T
    )
    translation = CARLA_TO_OPTICAL @ np.array(
        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=float
    )
    pose = np.empty((3, 4), dtype=float)
    pose[:, :3] = rotation_optical
    pose[:, 3] = translation
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    pose_csv = args.dataset.expanduser().resolve() / "pose_gt.csv"
    with pose_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = [str(index) for index in range(len(rows))]
    if [row["image_index"] for row in rows] != expected:
        raise SystemExit("pose_gt.csv image_index is not contiguous from zero")

    matrices = np.asarray([convert_row(row).reshape(-1) for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, matrices, fmt="%.9f")
    print(f"Converted {len(rows)} CARLA poses: {args.output.resolve()}")


if __name__ == "__main__":
    main()
