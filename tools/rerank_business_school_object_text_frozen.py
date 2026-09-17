#!/usr/bin/env python3
"""Frozen-query comparison of legacy and local-neighbour semantic scoring."""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rclpy

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent))

from slam.cognitive_math_model import CognitiveMathModel
from slam.human_slam_node import HuMemSLAMNode
from slam.types import KeyframeRecord, SceneRecord


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def make_record(node, frame, keyframe_id, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    started = time.perf_counter()
    objects = node.process_yolo_results(
        image, run_ocr=(keyframe_id % node.ocr_keyframe_interval == 0)
    )
    return KeyframeRecord(
        keyframe_id=keyframe_id,
        timestamp=float(frame),
        scene=SceneRecord(embedding=None),
        static_objects=objects,
        source_frame_id=frame,
    ), (time.perf_counter() - started) * 1000.0


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_path = args.source_run / "human_candidates.csv"
    keyframe_path = args.source_run / "orb_keyframes.csv"
    timestamps = [line.split()[0] for line in
                  (args.sequence / "times.txt").read_text().splitlines()
                  if line.strip()]

    def image_path(frame):
        if not 0 <= frame < len(timestamps):
            raise IndexError(f"Dataset frame {frame} outside timestamp table")
        return args.sequence / "undistorted_images/cam0" / f"{timestamps[frame]}.png"

    with keyframe_path.open(newline="", encoding="utf-8") as stream:
        keyframes = list(csv.DictReader(stream))
    frame_to_kf = {int(row["frame_id"]): int(row["keyframe_id"])
                   for row in keyframes}
    ordered_frames = [int(row["frame_id"]) for row in keyframes]
    frame_index = {frame: index for index, frame in enumerate(ordered_frames)}

    groups = defaultdict(dict)
    query_paths = {}
    query_kf = {}
    candidate_paths = {}
    with candidate_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            query = int(row["query_frame_id"])
            candidate = int(row["candidate_source_frame_id"])
            groups[query][candidate] = None
            query_paths[query] = image_path(query)
            query_kf[query] = int(row["query_reference_keyframe_id"] or query)
            candidate_paths[candidate] = image_path(candidate)



    rclpy.init(args=[
        "--ros-args", "--params-file", str(args.params),
        "-p", "use_scene:=false", "-p", "use_object:=true",
        "-p", "use_text:=true", "-p", "async_processing:=false",
    ])
    node = HuMemSLAMNode()
    records = {}
    inference_ms = []
    try:
        work = dict(candidate_paths)
        work.update(query_paths)
        for number, (frame, path) in enumerate(sorted(work.items()), start=1):
            keyframe_id = frame_to_kf.get(frame, query_kf.get(frame, frame))
            records[frame], latency = make_record(node, frame, keyframe_id, path)
            inference_ms.append(latency)
            if number % 100 == 0 or number == len(work):
                print(f"Inferred {number}/{len(work)} unique frames", flush=True)

        legacy = CognitiveMathModel(
            w_scene=node.w_scene, w_object=node.w_object, w_text=node.w_text,
            sigma_mask=node.sigma_mask, lambda_area=node.lambda_area,
            text_geom_threshold=node.text_geom_threshold,
            semantic_threshold=node.semantic_threshold,
            use_scene=False, use_object=True, use_text=True,
            text_conflict_floor=node.text_conflict_floor,
            relative_layout_weight=0.0,
            persistence_gain=0.0,
        )

        def priors(frame):
            index = frame_index.get(frame)
            if index is None:
                earlier = [value for value in ordered_frames if value < frame]
                return [records[value] for value in earlier[-2:][::-1]
                        if value in records]
            return [records[ordered_frames[item]]
                    for item in range(index - 1, max(-1, index - 3), -1)
                    if ordered_frames[item] in records]

        outputs = {name: [] for name in (
            "legacy_object", "current_object",
            "legacy_object_text", "current_object_text",
        )}
        ranking_times = defaultdict(list)
        for number, (query_frame, candidates) in enumerate(sorted(groups.items()), start=1):
            query = records[query_frame]
            enriched_query = node._semantic_neighborhood(query, priors(query_frame))
            scored = {name: [] for name in outputs}
            started = time.perf_counter()
            for candidate_frame in candidates:
                candidate = records[candidate_frame]
                enriched_candidate = node._semantic_neighborhood(
                    candidate, priors(candidate_frame)
                )
                scored["legacy_object"].append((
                    legacy.object_similarity(query, candidate), candidate_frame
                ))
                scored["current_object"].append((
                    node.matcher.object_similarity(enriched_query, enriched_candidate),
                    candidate_frame,
                ))
                scored["legacy_object_text"].append((
                    legacy.score_breakdown(query, candidate, [], [])["unified_score"],
                    candidate_frame,
                ))
                scored["current_object_text"].append((
                    node.matcher.score_breakdown(
                        enriched_query, enriched_candidate, [], []
                    )["unified_score"],
                    candidate_frame,
                ))
            elapsed = (time.perf_counter() - started) * 1000.0
            for name, values in scored.items():
                ranking_times[name].append(elapsed / len(scored))
                values.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                for rank, (score, candidate_frame) in enumerate(values, start=1):
                    outputs[name].append({
                        "query_frame_id": query_frame,
                        "rank": rank,
                        "candidate_source_frame_id": candidate_frame,
                        "score": score,
                    })
            if number % 25 == 0 or number == len(groups):
                print(f"Ranked {number}/{len(groups)} frozen queries", flush=True)

        for name, rows in outputs.items():
            path = args.output / f"{name}_candidates.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (args.output / "runtime.json").write_text(json.dumps({
            "unique_frames": len(records),
            "queries": len(groups),
            "mean_shared_inference_ms": float(np.mean(inference_ms)),
            "mean_ranking_ms": {
                name: float(np.mean(values))
                for name, values in ranking_times.items()
            },
            "source_run": str(args.source_run),
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
