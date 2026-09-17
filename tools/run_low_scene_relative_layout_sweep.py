#!/usr/bin/env python3
"""Controlled relative-layout sweep on one frozen low-scene VPR subset."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from itertools import product
from pathlib import Path

import rclpy

from slam.human_slam_node import HuMemSLAMNode
from run_external_vpr_comparison import dataset_definition
from run_humanslam_vpr_comparison import rank_query, semantic_record


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=(
        "business_fall", "carla_dense_fog_overcast",
        "carla_extreme_rain_fog", "garage_feb2021",
    ))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--params", type=Path,
                        default=ROOT.parent / "config/human_slam_params.yaml")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=(0.05, 0.10, 0.20, 0.30))
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=(0.15, 0.25, 0.35))
    return parser.parse_args()


def mean(values):
    return statistics.fmean(values) if values else None


def main():
    args = arguments()
    data = dataset_definition(args.dataset, args.max_queries)
    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=[
        "--ros-args", "--params-file", str(args.params),
        "-p", "fusion_mode:=scene_support",
        "-p", "use_scene:=true", "-p", "use_object:=true",
        "-p", "use_text:=true",
    ])
    node = HuMemSLAMNode()
    try:
        database = {
            frame: semantic_record(node, data["database_paths"][frame], frame)[0]
            for frame in data["database_frames"]
        }
        queries = {
            frame: semantic_record(node, data["query_paths"][frame], frame)[0]
            for frame in data["query_frames"]
        }
        variants = [(0.0, node.matcher.relative_layout_sigma)] + list(
            product(args.weights, args.sigmas)
        )
        all_rows = []
        baseline = {}
        summaries = []
        for weight, sigma in variants:
            node.matcher.relative_layout_weight = float(weight)
            node.matcher.relative_layout_sigma = float(sigma)
            rows = []
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
                row = {
                    "dataset": args.dataset, "layout_weight": weight,
                    "layout_sigma": sigma, "query_frame": query,
                    "candidate_count": len(candidates),
                    "rank1_frame": int(top[0][0].source_frame_id),
                    "rank1_score": float(top[0][1]),
                    "rank1_correct": int(labels[0]),
                    "first_correct_rank": first or "",
                    "recall_at_5": int(any(labels)), "ranking_ms": ranking_ms,
                    "scene_score": top[0][2].get("scene_score"),
                    "object_score": top[0][2].get("object_score"),
                    "text_score": top[0][2].get("text_score"),
                    "text_evidence": top[0][2].get("text_evidence"),
                    "support_score": top[0][2].get("support_score"),
                }
                rows.append(row); all_rows.append(row)
            if weight == 0.0:
                baseline = {row["query_frame"]: row for row in rows}
            beneficial = harmful = changes = 0
            reciprocal = []
            for row in rows:
                base = baseline.get(row["query_frame"], row)
                changed = row["rank1_frame"] != base["rank1_frame"]
                changes += changed
                beneficial += changed and not base["rank1_correct"] and row["rank1_correct"]
                harmful += changed and base["rank1_correct"] and not row["rank1_correct"]
                rank = row["first_correct_rank"]
                reciprocal.append(1.0 / int(rank) if rank else 0.0)
            summaries.append({
                "dataset": args.dataset, "layout_weight": weight,
                "layout_sigma": sigma,
                "queries": len(rows),
                "recall_at_1": mean([row["rank1_correct"] for row in rows]),
                "recall_at_5": mean([row["recall_at_5"] for row in rows]),
                "mrr": mean(reciprocal), "rank1_changes": changes,
                "beneficial_changes": beneficial, "harmful_changes": harmful,
                "ranking_mean_ms": mean([row["ranking_ms"] for row in rows]),
            })
        for name, rows in (("query_results.csv", all_rows), ("summary.csv", summaries)):
            with (output / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        best = max(summaries, key=lambda row: (
            row["recall_at_1"], row["recall_at_5"], row["mrr"],
            -row["harmful_changes"], -row["ranking_mean_ms"],
        ))
        (output / "summary.json").write_text(json.dumps({
            "dataset": args.dataset, "evidence_protocol": "identical frozen records",
            "deployed_configuration_changed": False,
            "baseline": summaries[0], "best": best, "variants": summaries,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": args.dataset, "baseline": summaries[0], "best": best}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
