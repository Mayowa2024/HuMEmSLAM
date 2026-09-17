#!/usr/bin/env python3
"""Evaluate HuMemSLAM object and object+text layers on labelled triplets."""

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

from slam.human_slam_node import HuMemSLAMNode
from slam.types import KeyframeRecord, SceneRecord


CONDITIONS = (
    "original", "dark_50", "dark_80", "blur_15", "blur_35",
    "combo_15_50", "combo_35_80",
)


def transform(image, condition):
    if condition == "original":
        return image.copy()
    if condition.startswith("dark_"):
        percent = int(condition.rsplit("_", 1)[1])
        return np.clip(image.astype(np.float32) * (1.0 - percent / 100.0),
                       0, 255).astype(np.uint8)
    if condition.startswith("blur_"):
        length = int(condition.rsplit("_", 1)[1])
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        return cv2.filter2D(image, -1, kernel)
    if condition.startswith("combo_"):
        _, blur, darkness = condition.split("_")
        return transform(transform(image, f"blur_{blur}"), f"dark_{darkness}")
    raise ValueError(f"Unknown condition: {condition}")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    return parser.parse_args()


def record_for(node, image, identifier, annotated_path=None):
    debug = image.copy() if annotated_path else None
    start = time.perf_counter()
    objects = node.process_yolo_results(image, run_ocr=True, debug_image=debug)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if annotated_path:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), debug)
    return KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=SceneRecord(embedding=None),
        static_objects=objects,
        source_frame_id=identifier,
    ), latency_ms


def texts(record):
    return [text.text for obj in record.static_objects for text in obj.texts]


def classes(record):
    return [obj.class_name for obj in record.static_objects]


def pair_scores(model, query, candidate):
    object_score = model.object_similarity(query, candidate)
    text_score, text_evidence = model.text_similarity(query, candidate)
    breakdown = model.score_breakdown(query, candidate, [], [])
    return {
        "object": object_score,
        "text": text_score,
        "text_evidence": text_evidence,
        "object_text": breakdown["unified_score"],
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_review(rows, output, limit=12):

    selected = sorted(rows, key=lambda row: (
        row["object_text_positive_ranked_first"],
        row["object_text_margin"],
    ))[:limit]
    width, height, footer = 420, 236, 76
    canvas = np.zeros((len(selected) * (height + footer), width * 3, 3), np.uint8)
    for row_index, row in enumerate(selected):
        y = row_index * (height + footer)
        entries = (
            ("QUERY", row["query_annotated"]),
            ("POSITIVE", row["positive_annotated"]),
            ("HARD NEGATIVE", row["negative_annotated"]),
        )
        for column, (label, path) in enumerate(entries):
            image = cv2.imread(path)
            image = cv2.resize(image, (width, height))
            x = column * width
            canvas[y:y + height, x:x + width] = image
            cv2.putText(canvas, label, (x + 8, y + height + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1,
                        cv2.LINE_AA)
        detail = (
            f"q={row['query_frame']} p={row['positive_frame']} "
            f"n={row['negative_frame']} {row['condition']} | "
            f"obj {row['positive_object_score']:.3f}/{row['negative_object_score']:.3f} | "
            f"obj+text {row['positive_object_text_score']:.3f}/"
            f"{row['negative_object_text_score']:.3f}"
        )
        cv2.putText(canvas, detail, (8, y + height + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (220, 220, 220), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    annotated = args.output / "annotated"
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        triplets = list(csv.DictReader(stream))

    rclpy.init(args=[
        "--ros-args", "--params-file", str(args.params),
        "-p", "use_scene:=false", "-p", "use_object:=true",
        "-p", "use_text:=true",
    ])
    node = HuMemSLAMNode()
    rows = []
    reference_cache = {}
    try:

        for triplet in triplets:
            for role in ("positive", "negative"):
                path = Path(triplet[f"{role}_path"])
                if path in reference_cache:
                    continue
                frame = int(triplet[f"{role}_frame"])
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(path)
                target = annotated / f"reference_{frame:06d}.png"
                reference_cache[path] = (*record_for(node, image, frame, target), target)

        identifier = 100000
        for triplet in triplets:
            base = cv2.imread(triplet["query_path"], cv2.IMREAD_COLOR)
            if base is None:
                raise FileNotFoundError(triplet["query_path"])
            positive, positive_ms, positive_debug = reference_cache[Path(triplet["positive_path"])]
            negative, negative_ms, negative_debug = reference_cache[Path(triplet["negative_path"])]
            for condition in args.conditions:
                identifier += 1
                query_debug = annotated / (
                    f"query_{int(triplet['query_frame']):06d}_{condition}.png"
                )
                query, query_ms = record_for(
                    node, transform(base, condition), identifier, query_debug
                )
                pos = pair_scores(node.matcher, query, positive)
                neg = pair_scores(node.matcher, query, negative)
                row = {
                    "condition": condition,
                    "query_frame": int(triplet["query_frame"]),
                    "positive_frame": int(triplet["positive_frame"]),
                    "negative_frame": int(triplet["negative_frame"]),
                    "positive_distance_m": float(triplet["positive_distance_m"]),
                    "negative_distance_m": float(triplet["negative_distance_m"]),
                    "query_object_count": len(query.static_objects),
                    "positive_object_count": len(positive.static_objects),
                    "negative_object_count": len(negative.static_objects),
                    "query_classes": json.dumps(classes(query)),
                    "positive_classes": json.dumps(classes(positive)),
                    "negative_classes": json.dumps(classes(negative)),
                    "query_texts": json.dumps(texts(query)),
                    "positive_texts": json.dumps(texts(positive)),
                    "negative_texts": json.dumps(texts(negative)),
                    "positive_object_score": pos["object"],
                    "negative_object_score": neg["object"],
                    "object_margin": pos["object"] - neg["object"],
                    "object_positive_ranked_first": int(pos["object"] > neg["object"]),
                    "positive_text_score": pos["text"],
                    "negative_text_score": neg["text"],
                    "positive_text_evidence": pos["text_evidence"],
                    "negative_text_evidence": neg["text_evidence"],
                    "positive_object_text_score": pos["object_text"],
                    "negative_object_text_score": neg["object_text"],
                    "object_text_margin": pos["object_text"] - neg["object_text"],
                    "object_text_positive_ranked_first": int(pos["object_text"] > neg["object_text"]),
                    "query_latency_ms": query_ms,
                    "positive_reference_latency_ms": positive_ms,
                    "negative_reference_latency_ms": negative_ms,
                    "query_annotated": str(query_debug),
                    "positive_annotated": str(positive_debug),
                    "negative_annotated": str(negative_debug),
                }
                rows.append(row)
                print(f"{len(rows):3d}/{len(triplets)*len(args.conditions)} "
                      f"q={row['query_frame']} {condition}: "
                      f"obj margin={row['object_margin']:+.3f}, "
                      f"obj+text margin={row['object_text_margin']:+.3f}", flush=True)

        write_csv(args.output / "pair_results.csv", rows)
        summaries = []
        for condition in ["all", *args.conditions]:
            subset = rows if condition == "all" else [
                row for row in rows if row["condition"] == condition
            ]
            summaries.append({
                "condition": condition,
                "comparisons": len(subset),
                "object_pairwise_accuracy": np.mean([row["object_positive_ranked_first"] for row in subset]),
                "object_mean_positive": np.mean([row["positive_object_score"] for row in subset]),
                "object_mean_negative": np.mean([row["negative_object_score"] for row in subset]),
                "object_mean_margin": np.mean([row["object_margin"] for row in subset]),
                "object_text_pairwise_accuracy": np.mean([row["object_text_positive_ranked_first"] for row in subset]),
                "object_text_mean_positive": np.mean([row["positive_object_text_score"] for row in subset]),
                "object_text_mean_negative": np.mean([row["negative_object_text_score"] for row in subset]),
                "object_text_mean_margin": np.mean([row["object_text_margin"] for row in subset]),
                "queries_with_objects": np.mean([row["query_object_count"] > 0 for row in subset]),
                "queries_with_text": np.mean([bool(json.loads(row["query_texts"])) for row in subset]),
                "mean_query_latency_ms": np.mean([row["query_latency_ms"] for row in subset]),
            })
        write_csv(args.output / "condition_summary.csv", summaries)
        render_review(rows, args.output / "lowest_margin_visual_review.png")
        (args.output / "run_metadata.json").write_text(json.dumps({
            "manifest": str(args.manifest),
            "params": str(args.params),
            "triplets": len(triplets),
            "conditions": args.conditions,
            "comparisons": len(rows),
            "scoring": "exact current HuMemSLAM object and query-driven object+text fusion",
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
