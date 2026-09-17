#!/usr/bin/env python3
"""Run full HuMemSLAM semantic ranking on the fixed external-VPR manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from slam.human_slam_node import HuMemSLAMNode
from types import SimpleNamespace

from run_external_vpr_comparison import dataset_definition, evenly_select
from slam.types import KeyframeRecord


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--params", type=Path, default=(
        ROOT.parent / "config/human_slam_params.yaml"
    ))
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument(
        "--candidate-top-k", type=int,
        help="Override the EigenPlaces prefilter size for controlled sweeps.",
    )
    parser.add_argument(
        "--fusion-mode", choices=("tri_layer", "scene_support"),
        help="Override semantic fusion for a controlled comparison.",
    )
    parser.add_argument("--ocr-backend", choices=["paddle", "tensorrt"])
    parser.add_argument("--ocr-detector-engine", type=Path)
    parser.add_argument("--ocr-recognizer-engine", type=Path)
    parser.add_argument("--ocr-recognizer-yaml", type=Path)
    return parser.parse_args()


def semantic_record(node, path, identifier):




    node._ocr_landmark_cache.clear()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    started = time.perf_counter()
    scene = node.run_scene_classifier(image.copy())
    objects = node.process_yolo_results(
        image.copy(), keyframe_id=identifier, run_ocr=True
    )
    latency = (time.perf_counter() - started) * 1000.0
    return KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=scene,
        static_objects=objects,
        source_frame_id=identifier,
    ), latency


def recall_at_full_precision(rows):
    wrong = [row["top1_score"] for row in rows if not row["top1_correct"]]
    threshold = max(wrong) + 1e-12 if wrong else -math.inf
    correct = sum(
        row["top1_correct"] and row["top1_score"] >= threshold for row in rows
    )
    return correct / len(rows), threshold


def rank_query(node, query_record, query, data, database):
    """Run the deployed scene prefilter and configured semantic fusion."""
    started = time.perf_counter()
    candidate_frames = [
        frame for frame in data["database_frames"] if frame <= query - 100
    ]
    scene_ranked = []
    for frame in candidate_frames:
        candidate = database[frame]
        score = node.matcher.scene_similarity(
            [query_record.scene], [candidate.scene]
        )
        scene_ranked.append((score, frame))
    scene_ranked.sort(reverse=True)
    selected_frames = [frame for _, frame in scene_ranked[:node.candidate_top_k]]
    ranked = node._rank_candidates(
        query_record,
        [query_record.scene],
        [database[frame] for frame in selected_frames],
        [[database[frame].scene] for frame in selected_frames],
    )
    return ranked, candidate_frames, (time.perf_counter() - started) * 1000.0


def main():
    args = arguments()
    data = dataset_definition(args.dataset, args.max_queries)
    output = args.output_root / args.dataset / "humanslam"
    output.mkdir(parents=True, exist_ok=True)
    ros_args = ["--ros-args", "--params-file", str(args.params)]
    if args.ocr_backend:
        ros_args += ["-p", f"ocr_backend:={args.ocr_backend}"]
    if args.candidate_top_k is not None:
        ros_args += ["-p", f"candidate_top_k:={args.candidate_top_k}"]
    if args.fusion_mode is not None:
        ros_args += ["-p", f"fusion_mode:={args.fusion_mode}"]
    for name, value in (
        ("ocr_detector_engine_path", args.ocr_detector_engine),
        ("ocr_recognizer_engine_path", args.ocr_recognizer_engine),
        ("ocr_recognizer_yaml_path", args.ocr_recognizer_yaml),
    ):
        if value:
            ros_args += ["-p", f"{name}:={value.resolve()}"]
    rclpy.init(args=ros_args)
    node = HuMemSLAMNode()
    try:
        database = {}
        database_latency = []
        for frame in data["database_frames"]:
            database[frame], latency = semantic_record(
                node, data["database_paths"][frame], frame
            )
            database_latency.append(latency)

        rows, query_latencies = [], []
        for query in data["query_frames"]:
            query_record, perception_ms = semantic_record(
                node, data["query_paths"][query], query
            )
            ranked, candidate_frames, ranking_ms = rank_query(
                node, query_record, query, data, database
            )
            labels = [
                data["positive"](query, int(item[0].source_frame_id))
                for item in ranked[:5]
            ]
            first = labels.index(True) + 1 if any(labels) else None
            top = ranked[0]
            rows.append({
                "query_frame": query,
                "candidate_count": len(candidate_frames),
                "rank1_frame": int(top[0].source_frame_id),
                "rank1_distance_m": float(np.linalg.norm(
                    data["positions"][query]
                    - data["positions"][int(top[0].source_frame_id)]
                )),
                "top1_score": float(top[1]),
                "top1_correct": int(labels[0]),
                "first_correct_rank": first or "",
                "recall_at_1": int(labels[0]),
                "recall_at_5": int(any(labels)),
                "perception_ms": perception_ms,
                "ranking_ms": ranking_ms,
                "total_ms": perception_ms + ranking_ms,
            })
            query_latencies.append(perception_ms + ranking_ms)

        timed_queries = evenly_select(data["query_frames"], min(30, len(data["query_frames"])))
        latency_rows = []
        if node.global_descriptor is None:
            raise RuntimeError("HuMemSLAM global descriptor is not enabled")
        warm_image = cv2.imread(str(data["query_paths"][timed_queries[0]]))
        for _ in range(10):
            node.global_descriptor.describe(warm_image)
        for repetition in range(1, args.latency_repetitions + 1):
            for query in timed_queries:

                image = cv2.imread(str(data["query_paths"][query]))
                descriptor_started = time.perf_counter()
                node.global_descriptor.describe(image)
                descriptor_ms = (time.perf_counter() - descriptor_started) * 1000.0
                query_record, perception_ms = semantic_record(
                    node, data["query_paths"][query], query
                )
                _, _, ranking_ms = rank_query(
                    node, query_record, query, data, database
                )
                latency_rows.append({
                    "repetition": repetition,
                    "query_frame": query,
                    "descriptor_ms": descriptor_ms,
                    "end_to_end_ms": perception_ms + ranking_ms,
                })

        ranks = [int(row["first_correct_rank"]) for row in rows
                 if row["first_correct_rank"] != ""]
        safe_recall, threshold = recall_at_full_precision(rows)
        descriptor_values = [row["descriptor_ms"] for row in latency_rows]
        latency_values = [row["end_to_end_ms"] for row in latency_rows]
        summary = {
            "method": "humanslam",
            "implementation": "full deployed semantic perception and fusion",
            "dataset": args.dataset,
            "sequence": data["sequence"],
            "database_frames": len(data["database_frames"]),
            "candidate_top_k": node.candidate_top_k,
            "fusion_mode": node.fusion_mode,
            "evaluated_queries": len(rows),
            "recall@1": float(np.mean([row["recall_at_1"] for row in rows])),
            "recall@5": float(np.mean([row["recall_at_5"] for row in rows])),
            "mrr": float(sum(1.0 / rank for rank in ranks) / len(rows)),
            "empirical_recall_at_100_percent_precision": safe_recall,
            "empirical_100_percent_precision_threshold": threshold,
            "latency_repetitions": args.latency_repetitions,
            "latency_queries_per_repetition": len(timed_queries),
            "latency_mean_ms": float(np.mean(latency_values)),
            "latency_median_ms": float(np.median(latency_values)),
            "latency_p95_ms": float(np.percentile(latency_values, 95)),
            "descriptor_latency_mean_ms": float(np.mean(descriptor_values)),
            "descriptor_latency_median_ms": float(np.median(descriptor_values)),
            "descriptor_latency_p95_ms": float(np.percentile(descriptor_values, 95)),
            "end_to_end_latency_mean_ms": float(np.mean(latency_values)),
            "end_to_end_latency_median_ms": float(np.median(latency_values)),
            "end_to_end_latency_p95_ms": float(np.percentile(latency_values, 95)),
            "latency_scope": (
                "decoded image -> scene/global descriptor, object, OCR, scene "
                f"prefilter and {node.fusion_mode} fusion -> top-5 ranking; "
                "image I/O and database construction excluded"
            ),
            "database_construction_mean_ms": float(np.mean(database_latency)),
            "warning": (
                "HuMemSLAM scope is broader than descriptor-forward latency. "
                "Recall at 100% precision is an empirical same-set ceiling. "
                "Temporal OCR cache state is cleared between frozen samples."
            ),
        }
        with (output / "query_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        with (output / "latency.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(latency_rows[0]))
            writer.writeheader(); writer.writerows(latency_rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
