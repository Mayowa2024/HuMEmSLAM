#!/usr/bin/env python3
"""Render labelled query/candidate contact sheets from a loop-pair CSV."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def panel(image, heading, detail, colour, size=(640, 400)):
    width, height = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height + 90, width, 3), dtype=np.uint8)
    canvas[:height] = resized
    cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), colour, 5)
    cv2.putText(canvas, heading, (12, height + 32),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (245, 245, 245), 2,
                cv2.LINE_AA)
    cv2.putText(canvas, detail, (12, height + 67),
                cv2.FONT_HERSHEY_SIMPLEX, .56, colour, 2, cv2.LINE_AA)
    return canvas


def main():
    args = arguments()
    images = sorted(
        (path for path in args.images.iterdir()
         if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda path: int(path.stem),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for index, row in enumerate(rows, start=1):
        query_id = int(row["query_frame_id"])
        candidate_id = int(row["candidate_frame_id"])
        query = cv2.imread(str(images[query_id]))
        candidate = cv2.imread(str(images[candidate_id]))
        if query is None or candidate is None:
            raise SystemExit(f"Missing frame for row {index}")
        correct = row["ground_truth_correct"] == "1"
        colour = (70, 220, 80) if correct else (50, 60, 240)
        status = "GT CORRECT" if correct else "GT FALSE"
        legacy = " | LEGACY RECONSTRUCTED" if row.get("legacy") == "1" else ""
        query_panel = panel(
            query, f"QUERY dataset frame {query_id}",
            f"{row['run']} | {row['method']}{legacy}", colour,
        )
        candidate_panel = panel(
            candidate, f"MATCHED dataset frame {candidate_id}",
            f"{status} | d3={float(row['distance_3d_m']):.2f} m | "
            f"angle={float(row['orientation_difference_deg']):.2f} deg",
            colour,
        )
        sheet = np.hstack((query_panel, candidate_panel))
        name = (f"{row['run']}_{row['method']}_closure_{int(row['closure']):02d}_"
                f"q{query_id}_c{candidate_id}_{'correct' if correct else 'false'}.png")
        cv2.imwrite(str(args.output / name), sheet)
        grouped.setdefault((row["run"], row["method"]), []).append(sheet)
    for (run, method), sheets in grouped.items():
        separator = np.zeros((12, sheets[0].shape[1], 3), dtype=np.uint8)
        overview_parts = []
        for sheet in sheets:
            if overview_parts:
                overview_parts.append(separator)
            overview_parts.append(sheet)
        cv2.imwrite(
            str(args.output / f"OVERVIEW_{run}_{method}.png"),
            np.vstack(overview_parts),
        )
    print(f"Rendered {len(rows)} loop pairs to {args.output}")


if __name__ == "__main__":
    main()
