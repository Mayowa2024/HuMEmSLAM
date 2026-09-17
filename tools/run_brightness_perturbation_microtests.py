#!/usr/bin/env python3
"""Measure HuMemSLAM self-match robustness under progressive darkening."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy

from slam.human_slam_node import HuMemSLAMNode
from slam.scene_categories import category_compatibility
from slam.types import KeyframeRecord


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-ids", nargs="+", required=True, type=int)
    parser.add_argument(
        "--brightness-levels",
        nargs="+",
        type=float,
        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args()


def process(node, image, identifier):
    started = time.perf_counter()
    scene_started = started
    scene = node.run_scene_classifier(image.copy())
    scene_finished = time.perf_counter()
    objects = node.process_yolo_results(image.copy(), run_ocr=True)
    finished = time.perf_counter()
    return KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=scene,
        static_objects=objects,
        source_frame_id=identifier,
    ), {
        "scene_ms": (scene_finished - scene_started) * 1000.0,
        "object_ocr_ms": (finished - scene_finished) * 1000.0,
        "total_perception_ms": (finished - started) * 1000.0,
    }


def text_count(record):
    return sum(len(obj.texts) for obj in record.static_objects)


def evidence(node, query):
    layers = ["scene"]
    denominator = node.matcher.w_scene
    if query.static_objects and node.matcher.use_object and node.matcher.w_object > 0:
        layers.append("object")
        denominator += node.matcher.w_object
    text_strength = node.matcher.text_evidence_strength(query)
    if text_strength > 0 and node.matcher.use_text and node.matcher.w_text > 0:
        layers.append("text")
        denominator += node.matcher.w_text * text_strength
    return layers, float(denominator), float(text_strength)


def annotated(image, lines):
    canvas = image.copy()
    overlay_height = 34 + 25 * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (14, 28 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def main():
    args = arguments()
    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    levels = sorted(set(args.brightness_levels), reverse=True)
    if not levels or any(level <= 0 or level > 1 for level in levels):
        raise SystemExit("Brightness levels must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    boundary_dir = args.output / "boundary_examples"
    boundary_dir.mkdir(exist_ok=True)

    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode()
    rows = []
    try:
        for run_number, frame_id in enumerate(args.frame_ids, start=1):
            if not 0 <= frame_id < len(image_paths):
                raise ValueError(f"Frame {frame_id} outside 0..{len(image_paths)-1}")
            path = image_paths[frame_id]
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError(f"Cannot read {path}")
            candidate, candidate_latency = process(node, original, run_number * 1000)

            run_rows = []
            for level_index, level in enumerate(levels, start=1):
                query_image = np.clip(original.astype(np.float32) * level, 0, 255).astype(np.uint8)
                query, query_latency = process(
                    node, query_image, run_number * 1000 + level_index
                )
                breakdown = node.matcher.score_breakdown(
                    query, candidate, [query.scene], [candidate.scene]
                )
                q_embedding = node.matcher._normalize(query.scene.embedding)
                c_embedding = node.matcher._normalize(candidate.scene.embedding)
                embedding_cosine = float(np.clip(np.dot(q_embedding, c_embedding), 0, 1))
                category_score = category_compatibility(
                    query.scene.category_distribution,
                    candidate.scene.category_distribution,
                )
                layers, denominator, text_strength = evidence(node, query)
                score = float(breakdown["unified_score"])
                passed = score >= args.threshold
                evidence_label = "weak_scene_only" if layers == ["scene"] else "strong_multi_layer"
                row = {
                    "run": run_number,
                    "dataset_frame_id": frame_id,
                    "image_path": str(path),
                    "brightness_retained": level,
                    "darkening_percent": (1.0 - level) * 100.0,
                    "threshold": args.threshold,
                    "passed": passed,
                    "evidence_label": evidence_label,
                    "active_layers": "+".join(layers),
                    "active_layer_count": len(layers),
                    "active_denominator": denominator,
                    "scene_embedding_cosine": embedding_cosine,
                    "scene_category_compatibility": category_score,
                    "scene_similarity_score": breakdown["scene_score"],
                    "object_similarity_score": breakdown["object_score"],
                    "text_similarity_score": breakdown["text_score"],
                    "text_evidence": breakdown["text_evidence"],
                    "query_text_strength": text_strength,
                    "unified_humanslam_score": score,
                    "query_object_count": len(query.static_objects),
                    "candidate_object_count": len(candidate.static_objects),
                    "query_text_count": text_count(query),
                    "candidate_text_count": text_count(candidate),
                    "candidate_scene_label": candidate.scene.label,
                    "query_scene_label": query.scene.label,
                    "candidate_total_ms": candidate_latency["total_perception_ms"],
                    "query_scene_ms": query_latency["scene_ms"],
                    "query_object_ocr_ms": query_latency["object_ocr_ms"],
                    "query_total_ms": query_latency["total_perception_ms"],
                }
                rows.append(row)
                run_rows.append((row, query_image))

            passing = [item for item in run_rows if item[0]["passed"]]
            failing = [item for item in run_rows if not item[0]["passed"]]
            last_pass = min(passing, key=lambda item: item[0]["brightness_retained"]) if passing else None
            first_fail = max(failing, key=lambda item: item[0]["brightness_retained"]) if failing else None
            run_dir = boundary_dir / f"run_{run_number:02d}_frame_{frame_id:06d}"
            run_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(run_dir / "original_candidate.png"), original)
            for name, item in (("last_passing_query.png", last_pass), ("first_failing_query.png", first_fail)):
                if item is None:
                    continue
                row, image = item
                lines = [
                    f"brightness={row['brightness_retained']:.2f} darkening={row['darkening_percent']:.0f}%",
                    f"score={row['unified_humanslam_score']:.3f} threshold={args.threshold:.2f} pass={row['passed']}",
                    f"evidence={row['evidence_label']} layers={row['active_layers']}",
                    f"objects query/candidate={row['query_object_count']}/{row['candidate_object_count']} text={row['query_text_count']}/{row['candidate_text_count']}",
                ]
                cv2.imwrite(str(run_dir / name), annotated(image, lines))

        fields = list(rows[0])
        with (args.output / "brightness_sweep.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        boundary_rows = []
        for run_number, frame_id in enumerate(args.frame_ids, start=1):
            subset = [row for row in rows if row["run"] == run_number]
            passing = [row for row in subset if row["passed"]]
            failing = [row for row in subset if not row["passed"]]
            last_pass = min(passing, key=lambda row: row["brightness_retained"]) if passing else None
            first_fail = max(failing, key=lambda row: row["brightness_retained"]) if failing else None
            boundary_rows.append({
                "run": run_number,
                "dataset_frame_id": frame_id,
                "last_passing_brightness": last_pass["brightness_retained"] if last_pass else "",
                "last_passing_score": last_pass["unified_humanslam_score"] if last_pass else "",
                "last_passing_evidence": last_pass["evidence_label"] if last_pass else "",
                "first_failing_brightness": first_fail["brightness_retained"] if first_fail else "",
                "first_failing_score": first_fail["unified_humanslam_score"] if first_fail else "",
                "first_failing_evidence": first_fail["evidence_label"] if first_fail else "",
            })
        with (args.output / "failure_boundaries.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(boundary_rows[0]))
            writer.writeheader()
            writer.writerows(boundary_rows)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        for run_number in range(1, len(args.frame_ids) + 1):
            subset = sorted(
                (row for row in rows if row["run"] == run_number),
                key=lambda row: row["brightness_retained"],
            )
            ax.plot(
                [row["brightness_retained"] for row in subset],
                [row["unified_humanslam_score"] for row in subset],
                alpha=0.28, linewidth=1.2,
            )
        matrix = np.asarray([
            [next(row["unified_humanslam_score"] for row in rows
                  if row["run"] == run and row["brightness_retained"] == level)
             for level in sorted(levels)]
            for run in range(1, len(args.frame_ids) + 1)
        ])
        ordered = np.asarray(sorted(levels))
        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        ax.plot(ordered, mean, color="black", linewidth=2.5, label="10-image mean")
        ax.fill_between(ordered, mean - std, mean + std, color="black", alpha=0.12, label="±1 SD")
        ax.axhline(args.threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold {args.threshold:.2f}")
        ax.set(title="HuMemSLAM self-match under progressive brightness reduction", xlabel="Brightness retained (lower = darker)", ylabel="Unified HuMemSLAM score", ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "score_vs_brightness.png", dpi=180)
        plt.close(fig)

        pass_rates, strong_rates, scene_only_rates = [], [], []
        for level in ordered:
            subset = [row for row in rows if row["brightness_retained"] == level]
            pass_rates.append(np.mean([row["passed"] for row in subset]))
            strong_rates.append(np.mean([row["passed"] and row["evidence_label"] == "strong_multi_layer" for row in subset]))
            scene_only_rates.append(np.mean([row["passed"] and row["evidence_label"] == "weak_scene_only" for row in subset]))
        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.plot(ordered, pass_rates, marker="o", linewidth=2, label="all passes")
        ax.plot(ordered, strong_rates, marker="o", linewidth=2, label="strong multi-layer passes")
        ax.plot(ordered, scene_only_rates, marker="o", linewidth=2, label="weak scene-only passes")
        ax.set(title="Pass rate and supporting evidence", xlabel="Brightness retained", ylabel="Fraction of 10 images", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "pass_rate_vs_brightness.png", dpi=180)
        plt.close(fig)

        summary = {
            "images": len(args.frame_ids),
            "comparisons": len(rows),
            "threshold": args.threshold,
            "brightness_levels": levels,
            "overall_passes": sum(row["passed"] for row in rows),
            "strong_multi_layer_passes": sum(row["passed"] and row["evidence_label"] == "strong_multi_layer" for row in rows),
            "weak_scene_only_passes": sum(row["passed"] and row["evidence_label"] == "weak_scene_only" for row in rows),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
