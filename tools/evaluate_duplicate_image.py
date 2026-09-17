#!/usr/bin/env python3
"""Run one image through HuMemSLAM twice and report its self-match score."""

import argparse
import json
import time
from pathlib import Path

import cv2
import rclpy

from slam.human_slam_node import HuMemSLAMNode
from slam.types import KeyframeRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=Path,
        help="Legacy identity test: use this image as both candidate and query.",
    )
    parser.add_argument("--candidate-image", type=Path)
    parser.add_argument("--query-image", type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image:
        candidate_path = query_path = args.image
    elif args.candidate_image and args.query_image:
        candidate_path, query_path = args.candidate_image, args.query_image
    else:
        parser.error("provide --image or both --candidate-image and --query-image")

    images = [
        cv2.imread(str(candidate_path), cv2.IMREAD_COLOR),
        cv2.imread(str(query_path), cv2.IMREAD_COLOR),
    ]
    for path, image in zip((candidate_path, query_path), images):
        if image is None:
            raise SystemExit(f"Cannot read {path}")

    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode()
    try:
        records = []
        inference_ms = []
        for keyframe_id, image in zip((1, 2), images):
            started = time.perf_counter()
            scene = node.run_scene_classifier(image.copy())
            objects = node.process_yolo_results(image.copy(), run_ocr=True)
            inference_ms.append((time.perf_counter() - started) * 1000.0)
            records.append(KeyframeRecord(
                keyframe_id=keyframe_id,
                timestamp=float(keyframe_id),
                scene=scene,
                static_objects=objects,
                source_frame_id=keyframe_id,
            ))
        result = node.matcher.score_breakdown(
            records[1], records[0], [records[1].scene], [records[0].scene]
        )
        report = {
            "candidate_image": str(candidate_path),
            "query_image": str(query_path),
            "global_descriptor": node.global_descriptor_name,
            "first": {
                "scene": records[0].scene.label,
                "scene_confidence": records[0].scene.confidence,
                "objects": [vars(item) for item in records[0].static_objects],
                "inference_ms": inference_ms[0],
            },
            "second": {
                "scene": records[1].scene.label,
                "scene_confidence": records[1].scene.confidence,
                "objects": [vars(item) for item in records[1].static_objects],
                "inference_ms": inference_ms[1],
            },
            "score": result,
        }

        encoded = json.dumps(report, indent=2, default=lambda value: vars(value))
        print(encoded)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
