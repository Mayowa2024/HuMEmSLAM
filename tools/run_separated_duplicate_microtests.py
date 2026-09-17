#!/usr/bin/env python3
"""Rank a revisited image after ten intervening consecutive frames."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from run_duplicate_image_microtests import object_assignment_audit, process
from slam.human_slam_node import HuMemSLAMNode
from slam.scene_categories import category_compatibility


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-frame-ids", nargs="+", required=True, type=int)
    parser.add_argument("--gap", type=int, default=10)
    return parser.parse_args()


def main():
    args = arguments()
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode()
    summaries = []
    try:
        for run_number, base_id in enumerate(args.base_frame_ids, start=1):
            run_dir = args.output / f"run_{run_number:02d}_base_{base_id:06d}"
            run_dir.mkdir(exist_ok=True)
            candidate_records = []
            perception_rows = []
            for offset in range(args.gap + 1):
                frame_id = base_id + offset
                image = cv2.imread(str(images[frame_id]), cv2.IMREAD_COLOR)
                record, latency = process(node, image, frame_id)
                candidate_records.append(record)
                perception_rows.append({"frame_id": frame_id, **latency})


            query_image = cv2.imread(str(images[base_id]), cv2.IMREAD_COLOR)
            query, query_latency = process(node, query_image, base_id)
            ranked = []
            for candidate in candidate_records:
                breakdown = node.matcher.score_breakdown(
                    query, candidate, [query.scene], [candidate.scene]
                )
                ranked.append((candidate, breakdown))
            ranked.sort(key=lambda item: item[1]["unified_score"], reverse=True)

            candidate_rows = []
            original_metrics = None
            for rank, (candidate, breakdown) in enumerate(ranked, start=1):
                cosine = float(np.clip(np.dot(
                    node.matcher._normalize(query.scene.embedding),
                    node.matcher._normalize(candidate.scene.embedding),
                ), 0.0, 1.0))
                compatibility = category_compatibility(
                    query.scene.category_distribution,
                    candidate.scene.category_distribution,
                )
                assignments, _ = object_assignment_audit(
                    node.matcher, query.static_objects, candidate.static_objects
                )
                row = {
                    "rank": rank,
                    "candidate_frame_id": candidate.source_frame_id,
                    "is_original_duplicate": candidate.source_frame_id == base_id,
                    "frame_separation": args.gap + 1 if candidate.source_frame_id == base_id
                                        else (base_id + args.gap + 1 - candidate.source_frame_id),
                    "scene_embedding_cosine": cosine,
                    "scene_category_compatibility": compatibility,
                    "scene_similarity_score": breakdown["scene_score"],
                    "query_object_count": len(query.static_objects),
                    "candidate_object_count": len(candidate.static_objects),
                    "matched_object_count": len(assignments),
                    "object_similarity_score": breakdown["object_score"],
                    "text_similarity_score": breakdown["text_score"],
                    "text_evidence": breakdown["text_evidence"],
                    "unified_humanslam_score": breakdown["unified_score"],
                    "image_path": str(images[candidate.source_frame_id]),
                }
                candidate_rows.append(row)
                if row["is_original_duplicate"]:
                    original_metrics = {**row, "object_assignments": assignments}

            with (run_dir / "candidate_ranking.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=candidate_rows[0].keys())
                writer.writeheader()
                writer.writerows(candidate_rows)

            report = {
                "run": run_number,
                "sequence_frame_ids": list(range(base_id, base_id + args.gap + 1))
                                      + [base_id],
                "query_frame_id": base_id,
                "query_image_path": str(images[base_id]),
                "candidate_min_keyframe_separation": node.candidate_min_separation,
                "eligible_in_current_production_filter": (
                    args.gap + 1 >= node.candidate_min_separation
                ),
                "original_duplicate": original_metrics,
                "top_candidate": candidate_rows[0],
                "top1_is_original": bool(candidate_rows[0]["is_original_duplicate"]),
                "top1_margin": (
                    candidate_rows[0]["unified_humanslam_score"]
                    - candidate_rows[1]["unified_humanslam_score"]
                ),
                "query_latency": query_latency,
                "sequence_perception_latency": perception_rows,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            summaries.append({
                "run": run_number,
                "base_frame_id": base_id,
                "sequence": f"{base_id}-{base_id + args.gap}, then {base_id}",
                "original_rank": original_metrics["rank"],
                "original_scene_embedding_cosine": original_metrics["scene_embedding_cosine"],
                "original_scene_similarity_score": original_metrics["scene_similarity_score"],
                "original_object_similarity_score": original_metrics["object_similarity_score"],
                "original_text_similarity_score": original_metrics["text_similarity_score"],
                "original_text_evidence": original_metrics["text_evidence"],
                "original_unified_score": original_metrics["unified_humanslam_score"],
                "top1_is_original": report["top1_is_original"],
                "top1_margin": report["top1_margin"],
                "production_filter_eligible": report["eligible_in_current_production_filter"],
            })

        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        top1 = sum(item["top1_is_original"] for item in summaries)
        scores = [item["original_unified_score"] for item in summaries]
        readme = f"""# HuMemSLAM duplicate revisit after ten intervening frames

Date: 2026-08-08

## Protocol

Each isolated run processes `base, base+1, ..., base+10, base`. The final base
image is ranked against all eleven stored observations. Layer inputs and the
complete candidate ranking are recorded for every run.

This directly tests scorer discrimination among highly similar consecutive
views. It bypasses the production temporal exclusion filter: separation is 11
observations while `candidate_min_keyframe_separation` is
{node.candidate_min_separation}.

## Aggregate result

- Runs: {len(summaries)}
- Original duplicate ranked top-1: {top1}/{len(summaries)}
- Mean original score: {np.mean(scores):.6f}
- Minimum original score: {np.min(scores):.6f}
- Maximum original score: {np.max(scores):.6f}

See `summary.csv`; each run directory contains `candidate_ranking.csv` and a
detailed `metrics.json`.
"""
        (args.output / "README.md").write_text(readme, encoding="utf-8")
        print(readme)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
