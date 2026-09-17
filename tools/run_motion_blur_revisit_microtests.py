#!/usr/bin/env python3
"""Compare HuMemSLAM and ORB on revisits under blur and darkness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy

from run_night_orb_humanslam_microtests import semantic_record
from run_night_revisit_microtests import (
    load_poses, orb_viewpoint_metrics, select_pairs,
)
from slam.human_slam_node import HuMemSLAMNode


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--query-start", type=int, default=828)
    parser.add_argument("--query-end", type=int, default=-1)
    parser.add_argument("--min-frame-separation", type=int, default=300)
    parser.add_argument("--max-distance-m", type=float, default=5.0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--kernels", nargs="+", type=int,
                        default=[0, 5, 9, 15, 25, 35])
    parser.add_argument("--angles", nargs="+", type=float,
                        default=[-10.0, 0.0, 10.0])
    parser.add_argument("--brightness-levels", nargs="+", type=float,
                        default=[1.0],
                        help="Retained brightness fractions (1.0 is unchanged)")
    parser.add_argument("--human-threshold", type=float, default=0.70)
    parser.add_argument("--orb-min-inliers", type=int, default=30)
    return parser.parse_args()


def motion_kernel(length: int, angle: float):
    if length <= 1:
        return np.ones((1, 1), dtype=np.float32)
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = (size - 1) / 2.0
    radius = (size - 1) / 2.0
    theta = np.radians(angle)
    dx, dy = radius * np.cos(theta), radius * np.sin(theta)
    cv2.line(kernel,
             (int(round(centre - dx)), int(round(centre - dy))),
             (int(round(centre + dx)), int(round(centre + dy))), 1, 1)
    result = kernel.astype(np.float32)
    return result / result.sum()


def apply_blur(image, length, angle):
    return cv2.filter2D(image, -1, motion_kernel(length, angle))


def apply_brightness(image, retained):
    """Reduce exposure in linear-light space to avoid a naive sRGB multiply."""
    value = image.astype(np.float32) / 255.0
    linear = np.where(value <= 0.04045, value / 12.92,
                      ((value + 0.055) / 1.055) ** 2.4)
    linear *= retained
    srgb = np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.clip(linear, 0, 1), 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def main():
    args = arguments()
    if any(kernel < 0 for kernel in args.kernels):
        raise SystemExit("Kernel lengths must be non-negative")
    if any(level <= 0 or level > 1 for level in args.brightness_levels):
        raise SystemExit("Brightness levels must be in the interval (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in args.images.iterdir()
                         if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    poses = load_poses(args.ground_truth)
    query_end = len(image_paths) - 1 if args.query_end < 0 else args.query_end
    pairs = select_pairs(
        poses, args.query_start, query_end, args.min_frame_separation,
        args.max_distance_m, args.max_rotation_deg, args.pairs)
    with (args.output / "selected_pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_frame", "candidate_frame", "distance_m", "rotation_deg"])
        writer.writerows(pairs)
    blur_conditions = [(0, 0.0)] + [
        (kernel, angle) for kernel in args.kernels if kernel > 1
        for angle in args.angles]
    conditions = [(kernel, angle, brightness)
                  for brightness in args.brightness_levels
                  for kernel, angle in blur_conditions]
    examples = args.output / "examples"; examples.mkdir(exist_ok=True)
    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode(); rows = []
    try:
        for pair_index, (query_id, candidate_id, distance, rotation) in enumerate(pairs):
            candidate_image = cv2.imread(str(image_paths[candidate_id]))
            query_image = cv2.imread(str(image_paths[query_id]))
            candidate, _ = semantic_record(node, candidate_image, pair_index * 10000)
            pair_dir = examples / f"q{query_id:06d}_c{candidate_id:06d}"
            pair_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(pair_dir / "candidate.png"), candidate_image)
            for condition_index, (kernel, angle, brightness) in enumerate(conditions):
                blurred = query_image if kernel == 0 else apply_blur(query_image, kernel, angle)
                perturbed = apply_brightness(blurred, brightness)
                query, latency = semantic_record(
                    node, perturbed, pair_index * 10000 + condition_index + 1)
                breakdown = node.matcher.score_breakdown(
                    query, candidate, [query.scene], [candidate.scene])
                score = float(breakdown["unified_score"])
                orb = orb_viewpoint_metrics(candidate_image, perturbed,
                                             args.orb_min_inliers)
                rows.append({
                    "pair": pair_index + 1, "query_frame": query_id,
                    "candidate_frame": candidate_id, "distance_m": distance,
                    "rotation_deg": rotation, "blur_kernel_px": kernel,
                    "blur_angle_deg": angle,
                    "brightness_retained": brightness,
                    "darkening_percent": (1.0 - brightness) * 100.0,
                    "human_score": score,
                    "human_pass": score > args.human_threshold,
                    "scene_score": breakdown["scene_score"],
                    "object_score": breakdown["object_score"],
                    "text_score": breakdown["text_score"],
                    "query_objects": len(query.static_objects),
                    "candidate_objects": len(candidate.static_objects),
                    "human_latency_ms": latency, **orb,
                    "humanslam_only_crossover": (
                        score > args.human_threshold and
                        not orb["orb_diagnostic_pass"]),
                })
                if kernel in (0, 15, 25, 35) and angle == 0:
                    cv2.imwrite(str(pair_dir / (
                        f"query_blur_{kernel:02d}px_dark_{(1-brightness)*100:02.0f}pct.png")),
                        perturbed)
        with (args.output / "paired_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = []
        for kernel, angle, brightness in conditions:
            subset = [row for row in rows if row["blur_kernel_px"] == kernel
                      and row["blur_angle_deg"] == angle
                      and row["brightness_retained"] == brightness]
            summary.append({
                "blur_kernel_px": kernel, "blur_angle_deg": angle,
                "brightness_retained": brightness,
                "darkening_percent": (1.0 - brightness) * 100.0,
                "pairs": len(subset),
                "human_score_mean": float(np.mean([r["human_score"] for r in subset])),
                "human_pass_rate": float(np.mean([r["human_pass"] for r in subset])),
                "orb_inliers_mean": float(np.mean([r["fundamental_inliers"] for r in subset])),
                "orb_pass_rate": float(np.mean([r["orb_diagnostic_pass"] for r in subset])),
                "fast12_retention_mean": float(np.mean([r["fast12_retention"] for r in subset])),
                "humanslam_only_crossovers": int(sum(
                    r["humanslam_only_crossover"] for r in subset)),
            })
        with (args.output / "condition_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        crossovers = [row for row in rows if row["humanslam_only_crossover"]]
        (args.output / "summary.json").write_text(json.dumps({
            "conditions": summary, "crossover_count": len(crossovers),
            "crossover_rows": crossovers,
            "warning": "ORB is an OpenCV FAST/ORB/fundamental-RANSAC screening proxy, not full ORB-SLAM3.",
        }, indent=2) + "\n")
        fig, left = plt.subplots(figsize=(11, 6.2))
        for brightness in args.brightness_levels:
            subset = [s for s in summary if s["blur_angle_deg"] == 0
                      and s["brightness_retained"] == brightness]
            left.plot([s["blur_kernel_px"] for s in subset],
                      [s["human_score_mean"] for s in subset], marker="o",
                      label=f"HuMemSLAM, {100*(1-brightness):.0f}% dark")
        left.axhline(args.human_threshold, color="black", linestyle="--",
                     label="Human threshold")
        left.set(xlabel="Motion-blur kernel (pixels)", ylabel="Mean HuMemSLAM score",
                 ylim=(0, 1)); left.grid(alpha=.2)
        right = left.twinx()
        zero_angle = [s for s in summary if s["blur_angle_deg"] == 0
                      and s["brightness_retained"] == min(args.brightness_levels)]
        right.plot([s["blur_kernel_px"] for s in zero_angle],
                   [s["orb_inliers_mean"] for s in zero_angle], color="tab:red",
                   marker="s", linewidth=2,
                   label=f"ORB/F inliers ({100*(1-min(args.brightness_levels)):.0f}% dark)")
        right.axhline(args.orb_min_inliers, color="tab:red", linestyle="--",
                      label="ORB diagnostic boundary")
        right.set_ylabel("Mean fundamental-matrix inliers")
        line1, label1 = left.get_legend_handles_labels()
        line2, label2 = right.get_legend_handles_labels()
        left.legend(line1 + line2, label1 + label2, loc="best")
        fig.tight_layout(); fig.savefig(args.output / "motion_blur_comparison.png", dpi=180)
        plt.close(fig)
        print(json.dumps({"conditions": summary,
                          "crossover_count": len(crossovers)}, indent=2))
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
