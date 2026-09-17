#!/usr/bin/env python3
"""Controlled scene-support ablation on frozen VPR queries and databases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from slam.human_slam_node import HuMemSLAMNode
from run_external_vpr_comparison import dataset_definition, evenly_select
from run_humanslam_vpr_comparison import semantic_record, rank_query


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "scene_only": (True, False, False),
    "scene_object": (True, True, False),
    "scene_text": (True, False, True),
    "full": (True, True, True),
}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--params", type=Path, default=ROOT.parent / "config/human_slam_params.yaml")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    return parser.parse_args()


def activate(node, flags):
    scene, objects, text = flags
    node.use_scene, node.use_object, node.use_text = scene, objects, text
    node.matcher.use_scene = scene
    node.matcher.use_object = objects
    node.matcher.use_text = text


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def main():
    options = args()
    data = dataset_definition(options.dataset, options.max_queries)
    output = options.output_root / options.dataset
    output.mkdir(parents=True, exist_ok=True)

    ros_args = [
        "--ros-args", "--params-file", str(options.params),
        "-p", "fusion_mode:=scene_support",
        "-p", "use_scene:=true", "-p", "use_object:=true", "-p", "use_text:=true",
    ]
    rclpy.init(args=ros_args)
    node = HuMemSLAMNode()
    try:


        database = {}
        for frame in data["database_frames"]:
            database[frame], _ = semantic_record(node, data["database_paths"][frame], frame)
        queries = {}
        for frame in data["query_frames"]:
            queries[frame], _ = semantic_record(node, data["query_paths"][frame], frame)

        results = {}
        for config, flags in CONFIGS.items():
            activate(node, flags)
            rows = []
            ranking_latencies = []
            for query in data["query_frames"]:
                ranked, candidates, ranking_ms = rank_query(
                    node, queries[query], query, data, database
                )
                top = ranked[:5]
                labels = [
                    data["positive"](query, int(item[0].source_frame_id))
                    for item in top
                ]
                first = labels.index(True) + 1 if any(labels) else None
                breakdown = top[0][2]
                rows.append({
                    "query_frame": query,
                    "candidate_count": len(candidates),
                    "rank1_frame": int(top[0][0].source_frame_id),
                    "rank1_score": float(top[0][1]),
                    "rank1_correct": int(labels[0]),
                    "first_correct_rank": first or "",
                    "recall_at_1": int(labels[0]),
                    "recall_at_5": int(any(labels)),
                    "scene_score": breakdown.get("scene_score"),
                    "object_score": breakdown.get("object_score"),
                    "text_score": breakdown.get("text_score"),
                    "text_evidence": breakdown.get("text_evidence"),
                    "support_score": breakdown.get("support_score"),
                    "ranking_ms": ranking_ms,
                })
                ranking_latencies.append(ranking_ms)
            results[config] = rows



        timed = evenly_select(data["query_frames"], min(30, len(data["query_frames"])))
        perception = {name: [] for name in CONFIGS}
        ranking_repeat = {name: [] for name in CONFIGS}
        for _ in range(options.latency_repetitions):
            for query in timed:
                image = cv2.imread(str(data["query_paths"][query]), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not decode {data['query_paths'][query]}")
                for config, flags in CONFIGS.items():
                    activate(node, flags)
                    node._ocr_landmark_cache.clear()
                    started = time.perf_counter()
                    node.run_scene_classifier(image.copy())
                    if flags[1] or flags[2]:
                        node.process_yolo_results(
                            image.copy(), keyframe_id=query, run_ocr=flags[2]
                        )
                    perception[config].append((time.perf_counter() - started) * 1000.0)
                    started = time.perf_counter()
                    rank_query(node, queries[query], query, data, database)
                    ranking_repeat[config].append((time.perf_counter() - started) * 1000.0)

        scene_rows = {int(row["query_frame"]): row for row in results["scene_only"]}
        summaries = []
        for config, rows in results.items():
            ranks = [int(row["first_correct_rank"]) for row in rows if row["first_correct_rank"] != ""]
            rank1_changes = beneficial = harmful = 0
            for row in rows:
                base = scene_rows[int(row["query_frame"])]
                if row["rank1_frame"] != base["rank1_frame"]:
                    rank1_changes += 1
                    beneficial += int(not base["rank1_correct"] and row["rank1_correct"])
                    harmful += int(base["rank1_correct"] and not row["rank1_correct"])
            p = perception[config]
            r = ranking_repeat[config]
            totals = [a + b for a, b in zip(p, r)]
            summary = {
                "dataset": options.dataset,
                "configuration": config,
                "database_frames": len(data["database_frames"]),
                "queries": len(rows),
                "recall@1": statistics.fmean(row["recall_at_1"] for row in rows),
                "recall@5": statistics.fmean(row["recall_at_5"] for row in rows),
                "mrr": sum(1.0 / rank for rank in ranks) / len(rows),
                "rank1_changes_vs_scene_only": rank1_changes,
                "beneficial_rank1_changes": beneficial,
                "harmful_rank1_changes": harmful,
                "perception_mean_ms": statistics.fmean(p),
                "ranking_mean_ms": statistics.fmean(r),
                "end_to_end_mean_ms": statistics.fmean(totals),
                "end_to_end_median_ms": statistics.median(totals),
                "end_to_end_p95_ms": percentile(totals, 95),
                "latency_samples": len(totals),
                "evidence_protocol": "identical frozen records, query frames and database frames",
            }
            summaries.append(summary)
            config_dir = output / config
            config_dir.mkdir(exist_ok=True)
            with (config_dir / "query_results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            (config_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
        (output / "manifest.json").write_text(json.dumps({
            "dataset": options.dataset,
            "sequence": data["sequence"],
            "database_frames": data["database_frames"],
            "query_frames": data["query_frames"],
            "configurations": CONFIGS,
        }, indent=2) + "\n")
        print(json.dumps(summaries, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
