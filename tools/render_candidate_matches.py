#!/usr/bin/env python3
"""Render HuMemSLAM query/candidate retrievals as an inspection video."""

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


W, H = 400, 200


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images", type=Path,
                        help="Dataset image directory; frame IDs index its sorted files")
    parser.add_argument("--semantic-frames", type=Path,
                        help="HuMemSLAM debug frames containing object annotations")
    parser.add_argument(
        "--frame-association", type=Path,
        help="trajectory_frame_ids.csv mapping internal to dataset frame IDs",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-ids", type=int, nargs="*",
                        help="Render only these query frame IDs")
    parser.add_argument("--image-output-dir", type=Path,
                        help="Also save one PNG contact sheet per query")
    return parser.parse_args()


def image_panel(path, title, lines=()):
    image = cv2.imread(str(path))
    if image is None:
        image = np.zeros((H, W, 3), dtype=np.uint8)
        lines = ("IMAGE UNAVAILABLE", *lines)
    else:
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H + 72, W, 3), dtype=np.uint8)
    canvas[:H] = image
    cv2.putText(canvas, title, (8, H + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for index, line in enumerate(lines[:2]):
        cv2.putText(canvas, line, (8, H + 43 + index * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 255), 1,
                    cv2.LINE_AA)
    return canvas


def main():
    args = arguments()
    frame_map = {}
    if args.frame_association and args.frame_association.exists():
        with args.frame_association.open(newline="") as stream:
            for association in csv.DictReader(stream):
                match = re.search(
                    r"dataset_frame=(\d+)",
                    association.get("source_frame_id", ""),
                )
                if match:
                    frame_map[int(association["frame_id"])] = int(match.group(1))
    images = None
    if args.images:
        images = sorted(path for path in args.images.iterdir()
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def frame_path(row, candidate=False):
        key = "candidate_source_frame_id" if candidate else "query_frame_id"
        recorded = "candidate_image_path" if candidate else "query_image_path"
        internal_index = int(row[key])
        if args.semantic_frames:
            annotated = args.semantic_frames / f"{internal_index:06d}.png"
            if annotated.exists():
                return annotated
        index = frame_map.get(internal_index, internal_index)
        if images is not None and 0 <= index < len(images):
            return images[index]
        return row[recorded]

    grouped = OrderedDict()
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if args.query_ids and int(row["query_frame_id"]) not in args.query_ids:
                continue
            grouped.setdefault(row["query_frame_id"], []).append(row)
    if not grouped:
        raise SystemExit("Candidate CSV contains no retrievals")

    width, height = W * 3, (H + 72) * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.image_output_dir:
        args.image_output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height))
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rank"]))
        shown = rows[:args.top_k]
        query_dataset_id = frame_map.get(int(query_id), int(query_id))
        query = image_panel(
            frame_path(shown[0]),
            f"QUERY dataset frame {query_dataset_id} (internal {query_id})",
        )
        panels = [query]
        for row in shown:
            score = float(
                row.get("effective_score")
                or row.get("raw_unified_score")
                or row.get("unified_score")
                or 0.0
            )
            parts = (
                f"score={score:.3f}  scene={row.get('scene_score', '')}",
                f"object={row.get('object_score', '')}  text={row.get('text_score', '')}",
            )
            candidate_internal_id = int(row["candidate_source_frame_id"])
            candidate_dataset_id = frame_map.get(
                candidate_internal_id, candidate_internal_id
            )
            panels.append(image_panel(
                frame_path(row, candidate=True),
                f"RANK {row['rank']}  dataset frame {candidate_dataset_id} "
                f"(internal {candidate_internal_id})",
                parts,
            ))
        while len(panels) < 6:
            panels.append(np.zeros_like(query))
        frame = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
        writer.write(frame)
        if args.image_output_dir:
            cv2.imwrite(str(args.image_output_dir / f"query_{int(query_id):06d}.png"), frame)
    writer.release()
    print(f"Rendered {len(grouped)} retrieval events to {args.output}")


if __name__ == "__main__":
    main()
