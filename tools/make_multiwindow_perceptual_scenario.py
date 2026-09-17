#!/usr/bin/env python3
"""Create a KITTI stereo scenario with appearance changes in named windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

from benchmark_global_descriptors import transform


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_window(value: str) -> tuple[int, int]:
    try:
        start, end = (int(item) for item in value.split(":", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("window must satisfy 0 <= START <= END")
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--condition", required=True,
        choices=("combo_15_60", "combo_15_80", "combo_35_80"),
    )
    parser.add_argument("--window", required=True, action="append", type=parse_window)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = []
    modified = 0
    for directory in ("image_0", "image_1"):
        source_dir = args.source / directory
        output_dir = args.output / directory
        output_dir.mkdir()
        images = sorted(path for path in source_dir.iterdir()
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if any(end >= len(images) for _, end in args.window):
            raise SystemExit(f"Window exceeds final frame {len(images)-1}")
        for frame_id, source in enumerate(images):
            destination = output_dir / source.name
            if any(start <= frame_id <= end for start, end in args.window):
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None or not cv2.imwrite(str(destination), transform(image, args.condition)):
                    raise RuntimeError(f"Could not transform {source}")
                modified += 1
            else:
                link_or_copy(source, destination)
        counts.append(len(images))
    if counts[0] != counts[1]:
        raise SystemExit(f"Stereo counts differ: {counts}")
    shutil.copy2(args.source / "times.txt", args.output / "times.txt")
    manifest = {
        "source": str(args.source.resolve()), "condition": args.condition,
        "windows_inclusive": args.window, "frame_count": counts[0],
        "modified_stereo_images": modified,
        "transform_source": "tools/benchmark_global_descriptors.py",
    }
    (args.output / "scenario.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
