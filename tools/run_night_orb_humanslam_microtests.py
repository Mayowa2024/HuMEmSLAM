#!/usr/bin/env python3
"""Paired HuMemSLAM and ORB/FAST self-match tests under synthetic night."""

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


PROFILES = {
    "day": dict(ev=0.0, shadow=1.0, noise=0.0, blur=1, vignette=0.0),
    "night_l1": dict(ev=-1.5, shadow=1.05, noise=1.5, blur=3, vignette=0.10),
    "night_l2": dict(ev=-2.5, shadow=1.15, noise=3.0, blur=5, vignette=0.18),
    "night_l3": dict(ev=-3.5, shadow=1.30, noise=5.0, blur=7, vignette=0.27),
    "night_l4": dict(ev=-4.5, shadow=1.50, noise=8.0, blur=11, vignette=0.36),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-ids", nargs="+", required=True, type=int)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=list(PROFILES))
    parser.add_argument("--human-threshold", type=float, default=0.70)
    parser.add_argument("--orb-min-inliers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def srgb_to_linear(image):
    value = image.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92,
                    ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image):
    value = np.clip(image, 0.0, 1.0)
    value = np.where(value <= 0.0031308, value * 12.92,
                     1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


def make_night(image, profile, seed):
    if profile["ev"] == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    linear = srgb_to_linear(image)
    linear = np.power(np.clip(linear, 0, 1), profile["shadow"])
    linear *= 2.0 ** profile["ev"]
    height, width = linear.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    radius = np.clip((xx * xx + yy * yy) / 2.0, 0, 1)
    linear *= (1.0 - profile["vignette"] * radius)[..., None]

    linear *= np.asarray([1.08, 1.00, 0.88], dtype=np.float32)
    image_night = linear_to_srgb(linear).astype(np.float32)
    signal_sigma = np.sqrt(np.maximum(image_night, 0.0)) * profile["noise"] * 0.18
    read_sigma = profile["noise"]
    image_night += rng.normal(0.0, signal_sigma + read_sigma, image_night.shape)
    image_night = np.clip(image_night, 0, 255).astype(np.uint8)
    kernel = profile["blur"]
    if kernel > 1:
        motion = np.zeros((kernel, kernel), dtype=np.float32)
        motion[kernel // 2, :] = 1.0 / kernel
        image_night = cv2.filter2D(image_night, -1, motion)
    return image_night


def semantic_record(node, image, identifier):
    start = time.perf_counter()
    scene = node.run_scene_classifier(image.copy())
    objects = node.process_yolo_results(image.copy(), run_ocr=True)
    return KeyframeRecord(
        keyframe_id=identifier, timestamp=float(identifier), scene=scene,
        static_objects=objects, source_frame_id=identifier,
    ), (time.perf_counter() - start) * 1000.0


def orb_metrics(candidate, query, min_inliers):
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    fast12 = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    fast7 = cv2.FastFeatureDetector_create(threshold=7, nonmaxSuppression=True)
    candidate_fast12 = len(fast12.detect(candidate_gray, None))
    query_fast12 = len(fast12.detect(query_gray, None))
    candidate_fast7 = len(fast7.detect(candidate_gray, None))
    query_fast7 = len(fast7.detect(query_gray, None))
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                         fastThreshold=7)
    candidate_kp, candidate_desc = orb.detectAndCompute(candidate_gray, None)
    query_kp, query_desc = orb.detectAndCompute(query_gray, None)
    good = []
    if candidate_desc is not None and query_desc is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            query_desc, candidate_desc, k=2)
        good = [first for pair in matches if len(pair) == 2
                for first, second in [pair] if first.distance < 0.75 * second.distance]
    inliers = 0
    if len(good) >= 4:
        query_points = np.float32([query_kp[m.queryIdx].pt for m in good])
        candidate_points = np.float32([candidate_kp[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(query_points, candidate_points,
                                     cv2.RANSAC, 3.0)
        inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / len(good) if good else 0.0
    return {
        "candidate_fast12": candidate_fast12, "query_fast12": query_fast12,
        "fast12_retention": query_fast12 / candidate_fast12 if candidate_fast12 else 0.0,
        "candidate_fast7": candidate_fast7, "query_fast7": query_fast7,
        "fast7_retention": query_fast7 / candidate_fast7 if candidate_fast7 else 0.0,
        "candidate_orb_keypoints": len(candidate_kp),
        "query_orb_keypoints": len(query_kp), "ratio_matches": len(good),
        "ransac_inliers": inliers, "ransac_inlier_ratio": inlier_ratio,

        "orb_diagnostic_pass": inliers >= min_inliers and inlier_ratio >= 0.25,
    }


def main():
    args = arguments()
    paths = sorted(path for path in args.images.iterdir()
                   if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    args.output.mkdir(parents=True, exist_ok=True)
    examples = args.output / "examples"
    examples.mkdir(exist_ok=True)
    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode()
    rows = []
    try:
        for run, frame_id in enumerate(args.frame_ids, 1):
            if not 0 <= frame_id < len(paths):
                raise ValueError(f"Frame {frame_id} outside 0..{len(paths)-1}")
            candidate_image = cv2.imread(str(paths[frame_id]))
            if candidate_image is None:
                raise ValueError(f"Cannot read {paths[frame_id]}")
            candidate, candidate_ms = semantic_record(node, candidate_image, run * 1000)
            frame_dir = examples / f"frame_{frame_id:06d}"
            frame_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(frame_dir / "day_candidate.png"), candidate_image)
            for index, name in enumerate(args.profiles):
                query_image = make_night(
                    candidate_image, PROFILES[name],
                    args.seed + frame_id * 100 + index)
                query, query_ms = semantic_record(node, query_image,
                                                  run * 1000 + index + 1)
                breakdown = node.matcher.score_breakdown(
                    query, candidate, [query.scene], [candidate.scene])
                human_score = float(breakdown["unified_score"])
                orb = orb_metrics(candidate_image, query_image,
                                  args.orb_min_inliers)
                q_embedding = node.matcher._normalize(query.scene.embedding)
                c_embedding = node.matcher._normalize(candidate.scene.embedding)
                row = {
                    "run": run, "dataset_frame_id": frame_id,
                    "image_path": str(paths[frame_id]), "profile": name,
                    **PROFILES[name], "human_threshold": args.human_threshold,
                    "human_score": human_score,
                    "human_pass": human_score > args.human_threshold,
                    "scene_score": breakdown["scene_score"],
                    "scene_embedding_cosine": float(np.clip(
                        np.dot(q_embedding, c_embedding), 0, 1)),
                    "scene_category_compatibility": category_compatibility(
                        query.scene.category_distribution,
                        candidate.scene.category_distribution),
                    "object_score": breakdown["object_score"],
                    "text_score": breakdown["text_score"],
                    "query_objects": len(query.static_objects),
                    "candidate_objects": len(candidate.static_objects),
                    "query_scene": query.scene.label,
                    "candidate_scene": candidate.scene.label,
                    "candidate_human_ms": candidate_ms,
                    "query_human_ms": query_ms, **orb,
                }
                rows.append(row)
                cv2.imwrite(str(frame_dir / f"{name}.png"), query_image)
        with (args.output / "paired_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = []
        for name in args.profiles:
            subset = [row for row in rows if row["profile"] == name]
            summary.append({
                "profile": name, "images": len(subset),
                "human_pass_rate": float(np.mean([r["human_pass"] for r in subset])),
                "human_score_mean": float(np.mean([r["human_score"] for r in subset])),
                "orb_pass_rate": float(np.mean([r["orb_diagnostic_pass"] for r in subset])),
                "orb_inliers_mean": float(np.mean([r["ransac_inliers"] for r in subset])),
                "fast12_retention_mean": float(np.mean([r["fast12_retention"] for r in subset])),
            })
        with (args.output / "profile_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        (args.output / "summary.json").write_text(json.dumps({
            "method": "duplicated daylight candidate versus synthetic-night query",
            "profiles": summary,
            "warning": "ORB pass is an OpenCV ORB/Hamming/RANSAC screening proxy, not full ORB-SLAM3 Sim3 verification.",
        }, indent=2) + "\n")
        x = np.arange(len(summary))
        fig, left = plt.subplots(figsize=(10, 5.8))
        left.plot(x, [r["human_score_mean"] for r in summary], marker="o",
                  linewidth=2, label="HuMemSLAM score")
        left.axhline(args.human_threshold, color="tab:blue", linestyle="--",
                     alpha=.7, label="HuMemSLAM threshold")
        left.set_ylabel("HuMemSLAM score")
        left.set_ylim(0, 1.02)
        right = left.twinx()
        right.plot(x, [r["orb_inliers_mean"] for r in summary], marker="s",
                   color="tab:red", linewidth=2, label="ORB RANSAC inliers")
        right.axhline(args.orb_min_inliers, color="tab:red", linestyle="--",
                      alpha=.7, label="ORB diagnostic boundary")
        right.set_ylabel("ORB geometric inliers")
        left.set_xticks(x, [r["profile"] for r in summary])
        left.set_xlabel("Synthetic-night profile")
        left.grid(alpha=.2)
        lines1, labels1 = left.get_legend_handles_labels()
        lines2, labels2 = right.get_legend_handles_labels()
        left.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout(); fig.savefig(args.output / "humanslam_vs_orb.png", dpi=180)
        plt.close(fig)
        print(json.dumps(summary, indent=2))
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
