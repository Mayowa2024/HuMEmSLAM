#!/usr/bin/env python3
"""Run isolated HuMemSLAM duplicate-image identity microtests."""

import argparse
import csv
import json
import math
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import rclpy
from scipy.optimize import linear_sum_assignment

from slam.human_slam_node import HuMemSLAMNode
from slam.scene_categories import category_compatibility
from slam.types import KeyframeRecord


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-ids", nargs="+", required=True, type=int)
    return parser.parse_args()


def object_assignment_audit(model, query_objects, candidate_objects):
    assignments = []
    matched_total = 0.0
    for class_name in sorted({item.class_name for item in query_objects}):
        query_class = [item for item in query_objects if item.class_name == class_name]
        candidate_class = [item for item in candidate_objects if item.class_name == class_name]
        if not candidate_class:
            continue
        scores = np.asarray([
            [model.object_match_score(query, candidate)
             for candidate in candidate_class]
            for query in query_class
        ], dtype=np.float64)
        q_indices, c_indices = linear_sum_assignment(scores, maximize=True)
        for q_index, c_index in zip(q_indices, c_indices):
            query = query_class[q_index]
            candidate = candidate_class[c_index]
            spatial = model.spatial_similarity(query, candidate)
            reliability = math.sqrt(query.seg_conf * candidate.seg_conf)
            contribution = float(scores[q_index, c_index])
            matched_total += contribution
            assignments.append({
                "class_name": class_name,
                "query_confidence": query.seg_conf,
                "candidate_confidence": candidate.seg_conf,
                "confidence_reliability": reliability,
                "query_centroid": [query.x_centroid, query.y_centroid],
                "candidate_centroid": [candidate.x_centroid, candidate.y_centroid],
                "query_area": query.area,
                "candidate_area": candidate.area,
                "mask_distance": model.mask_distance(query, candidate),
                "spatial_similarity": spatial,
                "object_match_contribution": contribution,
            })
    return assignments, matched_total


def process(node, image, identifier):
    started = time.perf_counter()
    scene_started = started
    scene = node.run_scene_classifier(image.copy())
    scene_finished = time.perf_counter()
    objects = node.process_yolo_results(image.copy(), run_ocr=True)
    finished = time.perf_counter()
    record = KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=scene,
        static_objects=objects,
        source_frame_id=identifier,
    )
    return record, {
        "scene_ms": (scene_finished - scene_started) * 1000.0,
        "object_ocr_ms": (finished - scene_finished) * 1000.0,
        "total_perception_ms": (finished - started) * 1000.0,
    }


def main():
    args = arguments()
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode()
    reports = []
    try:
        for run_number, frame_id in enumerate(args.frame_ids, start=1):
            if not 0 <= frame_id < len(images):
                raise ValueError(f"Frame {frame_id} outside 0..{len(images)-1}")
            image_path = images[frame_id]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            first, first_latency = process(node, image, run_number * 2 - 1)
            second, second_latency = process(node, image, run_number * 2)

            q_emb = node.matcher._normalize(second.scene.embedding)
            c_emb = node.matcher._normalize(first.scene.embedding)
            cosine = float(np.clip(np.dot(q_emb, c_emb), 0.0, 1.0))
            compatibility = category_compatibility(
                second.scene.category_distribution,
                first.scene.category_distribution,
            )
            modulation = (
                (1.0 + node.scene_category_weight * compatibility)
                / (1.0 + node.scene_category_weight)
            )
            score = node.matcher.score_breakdown(
                second, first, [second.scene], [first.scene]
            )
            assignments, matched_total = object_assignment_audit(
                node.matcher, second.static_objects, first.static_objects
            )
            report = {
                "run": run_number,
                "dataset_frame_id": frame_id,
                "image_path": str(image_path),
                "scene_label_first": first.scene.label,
                "scene_label_second": second.scene.label,
                "scene_embedding_cosine": cosine,
                "scene_category_first": first.scene.category_distribution,
                "scene_category_second": second.scene.category_distribution,
                "scene_category_compatibility": compatibility,
                "scene_category_modulation": modulation,
                "scene_similarity_score": score["scene_score"],
                "query_object_count": len(second.static_objects),
                "candidate_object_count": len(first.static_objects),
                "matched_object_count": len(assignments),
                "matched_object_contribution_sum": matched_total,
                "object_assignments": assignments,
                "query_objects": [
                    {
                        "class_name": obj.class_name,
                        "confidence": obj.seg_conf,
                        "centroid": [obj.x_centroid, obj.y_centroid],
                        "area": obj.area,
                        "texts": [
                            {"text": text.text, "confidence": text.conf}
                            for text in obj.texts
                        ],
                    }
                    for obj in second.static_objects
                ],
                "candidate_objects": [
                    {
                        "class_name": obj.class_name,
                        "confidence": obj.seg_conf,
                        "centroid": [obj.x_centroid, obj.y_centroid],
                        "area": obj.area,
                        "texts": [
                            {"text": text.text, "confidence": text.conf}
                            for text in obj.texts
                        ],
                    }
                    for obj in first.static_objects
                ],
                "object_similarity_score": score["object_score"],
                "text_similarity_score": score["text_score"],
                "text_evidence": score["text_evidence"],
                "unified_humanslam_score": score["unified_score"],
                "first_pass_latency": first_latency,
                "second_pass_latency": second_latency,
            }
            reports.append(report)
            run_dir = args.output / f"run_{run_number:02d}_frame_{frame_id:06d}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            cv2.imwrite(str(run_dir / "input_image.png"), image)

        fields = [
            "run", "dataset_frame_id", "image_path", "scene_label_first",
            "scene_label_second", "scene_embedding_cosine",
            "scene_category_compatibility", "scene_category_modulation",
            "scene_similarity_score", "query_object_count",
            "candidate_object_count", "matched_object_count",
            "object_similarity_score", "text_similarity_score",
            "text_evidence", "unified_humanslam_score",
            "first_total_ms", "second_total_ms",
        ]
        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for report in reports:
                row = {key: report[key] for key in fields
                       if key not in {"first_total_ms", "second_total_ms"}}
                row["first_total_ms"] = report["first_pass_latency"]["total_perception_ms"]
                row["second_total_ms"] = report["second_pass_latency"]["total_perception_ms"]
                writer.writerow(row)

        scores = [item["unified_humanslam_score"] for item in reports]
        readme = f"""# HuMemSLAM duplicate-image identity microtest

Date: {date.today().isoformat()}

## Purpose

Measure HuMemSLAM's practical identity-match ceiling and determinism. Each of
10 source images was independently processed twice and the two semantic records
were compared. This is a controlled component test, not a localisation test.

## Configuration

- Image directory: `{args.images}`
- Frames: {', '.join(map(str, args.frame_ids))}
- Scene: Places365 TensorRT embedding and grouped category context
- Objects: HuMemSLAM Mapillary YOLO segmentation TensorRT engine
- OCR: GPU PaddleOCR
- Matching: exact-class, one-to-one object assignment

## Aggregate identity results

- Runs: {len(reports)}
- Mean HuMemSLAM score: {np.mean(scores):.6f}
- Minimum HuMemSLAM score: {np.min(scores):.6f}
- Maximum HuMemSLAM score: {np.max(scores):.6f}
- Standard deviation: {np.std(scores):.6f}

`summary.csv` contains comparable metrics. Each `run_*` directory contains the
exact input image and a detailed `metrics.json`, including every object pair's
confidence, mask distance, spatial similarity and score contribution.
"""
        (args.output / "README.md").write_text(readme, encoding="utf-8")
        print(readme)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
