#!/usr/bin/env python3
"""Measure whether HuMemSLAM's correct place survives its VPR prefilter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from run_external_vpr_comparison import dataset_definition
from slam.global_place_descriptor import DescriptorSpec, TensorRTGlobalDescriptor


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="carla_extreme_rain_fog")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 25, 50, 100])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--database-augmentations", nargs="*", default=[],
        choices=("fog_light", "fog_medium", "low_contrast"),
        help="Offline-only reference variants; query inference is unchanged.",
    )
    parser.add_argument(
        "--descriptor-config", type=Path,
        default=ROOT / "weights/global_descriptors/eigenplaces_r18_512.json",
    )
    args = parser.parse_args()

    data = dataset_definition(args.dataset, args.max_queries)
    runtime = TensorRTGlobalDescriptor(
        DescriptorSpec.from_json(args.descriptor_config)
    )
    frames = sorted(set(data["database_frames"] + data["query_frames"]))
    query_set = set(data["query_frames"])
    descriptors = {}
    database_variants = {}
    for frame in frames:
        path = data["query_paths"][frame] if frame in query_set else data["database_paths"][frame]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        descriptors[frame] = runtime.describe(image)
        if frame in data["database_frames"]:
            variants = [descriptors[frame]]
            for augmentation in args.database_augmentations:
                if augmentation == "fog_light":
                    altered = cv2.addWeighted(
                        image, 0.70, np.full_like(image, 210), 0.30, 0
                    )
                elif augmentation == "fog_medium":
                    altered = cv2.addWeighted(
                        image, 0.45, np.full_like(image, 210), 0.55, 0
                    )
                else:
                    altered = cv2.convertScaleAbs(image, alpha=0.35, beta=75)
                variants.append(runtime.describe(altered))
            database_variants[frame] = variants

    rows = []
    for query in data["query_frames"]:
        candidates = [f for f in data["database_frames"] if f <= query - 100]


        scores = np.asarray([
            max(float(value @ descriptors[query]) for value in database_variants[f])
            for f in candidates
        ])
        order = np.argsort(-scores)
        positive_ranks = [
            rank for rank, index in enumerate(order, 1)
            if data["positive"](query, candidates[index])
        ]
        best_rank = min(positive_ranks) if positive_ranks else None
        row = {
            "query_frame": query,
            "best_positive_rank": best_rank or "",
            "best_positive_score": (
                float(scores[order[best_rank - 1]]) if best_rank else ""
            ),
        }
        for k in args.top_k:
            row[f"positive_in_top{k}"] = int(best_rank is not None and best_rank <= k)
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "query_prefilter_ranks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid_ranks = [int(r["best_positive_rank"]) for r in rows if r["best_positive_rank"] != ""]
    summary = {
        "dataset": args.dataset,
        "queries": len(rows),
        "database_frames": len(data["database_frames"]),
        "database_augmentations": args.database_augmentations,
        "descriptors_per_database_frame": 1 + len(args.database_augmentations),
        "coverage": {
            f"top{k}": float(np.mean([r[f"positive_in_top{k}"] for r in rows]))
            for k in args.top_k
        },
        "median_best_positive_rank": float(np.median(valid_ranks)),
        "p95_best_positive_rank": float(np.percentile(valid_ranks, 95)),
        "queries_without_any_gt_positive_rank": len(rows) - len(valid_ranks),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
