#!/usr/bin/env python3
"""Compare HuMemSLAM and ORB on real revisits transformed to synthetic night."""

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

from run_night_orb_humanslam_microtests import (
    PROFILES, make_night, semantic_record,
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
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=list(PROFILES))
    parser.add_argument("--human-threshold", type=float, default=0.70)
    parser.add_argument("--orb-min-inliers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_poses(path):
    poses = []
    for line in path.read_text().splitlines():
        values = np.asarray([float(v) for v in line.split()])
        if values.size != 12:
            raise ValueError("Ground truth must contain KITTI 3x4 poses")
        pose = np.eye(4); pose[:3] = values.reshape(3, 4); poses.append(pose)
    return poses


def rotation_deg(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def select_pairs(poses, start, end, minimum, max_distance, max_rotation, count):
    eligible = []
    for query in range(start, min(end + 1, len(poses))):
        best = None
        for candidate in range(0, query - minimum + 1):
            distance = float(np.linalg.norm(
                poses[query][:3, 3] - poses[candidate][:3, 3]))
            if distance > max_distance:
                continue
            angle = rotation_deg(poses[query], poses[candidate])
            if angle > max_rotation:
                continue
            quality = distance + 0.05 * angle
            if best is None or quality < best[0]:
                best = (quality, candidate, distance, angle)
        if best:
            eligible.append((query, best[1], best[2], best[3]))
    if not eligible:
        raise SystemExit("No ground-truth revisit pairs satisfy the thresholds")
    indices = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    return [eligible[index] for index in sorted(set(indices))]


def add_nonuniform_lighting(image, severity, seed):
    if severity <= 0:
        return image
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    value = image.astype(np.float32)

    shadow = np.ones((height, width), dtype=np.float32)
    for _ in range(2):
        cx, cy = rng.uniform(0, width), rng.uniform(height * .15, height)
        sx, sy = rng.uniform(width * .18, width * .42), rng.uniform(height * .18, height * .5)
        gaussian = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
        shadow *= 1.0 - severity * rng.uniform(.25, .5) * gaussian
    value *= shadow[..., None]

    cx, cy = rng.uniform(width * .15, width * .85), rng.uniform(0, height * .42)
    sigma = rng.uniform(width * .025, width * .07)
    bloom = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)))
    colour = np.asarray([70, 150, 255], dtype=np.float32)
    value += severity * bloom[..., None] * colour
    core = ((xx - cx) ** 2 + (yy - cy) ** 2) < (sigma * .20) ** 2
    value[core] = 255
    return np.clip(value, 0, 255).astype(np.uint8)


def orb_viewpoint_metrics(candidate, query, minimum):
    cg = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    qg = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    fast12 = cv2.FastFeatureDetector_create(12, True)
    fast7 = cv2.FastFeatureDetector_create(7, True)
    c12, q12 = len(fast12.detect(cg)), len(fast12.detect(qg))
    c7, q7 = len(fast7.detect(cg)), len(fast7.detect(qg))
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                         fastThreshold=7)
    ckp, cd = orb.detectAndCompute(cg, None)
    qkp, qd = orb.detectAndCompute(qg, None)
    good = []
    if cd is not None and qd is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(qd, cd, k=2):
            if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance:
                good.append(pair[0])
    inliers = 0
    if len(good) >= 8:
        qp = np.float32([qkp[m.queryIdx].pt for m in good])
        cp = np.float32([ckp[m.trainIdx].pt for m in good])
        _, mask = cv2.findFundamentalMat(qp, cp, cv2.FM_RANSAC, 1.5, .99)
        inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / len(good) if good else 0.0
    return {
        "candidate_fast12": c12, "query_fast12": q12,
        "fast12_retention": q12 / c12 if c12 else 0,
        "candidate_fast7": c7, "query_fast7": q7,
        "fast7_retention": q7 / c7 if c7 else 0,
        "candidate_orb_keypoints": len(ckp), "query_orb_keypoints": len(qkp),
        "ratio_matches": len(good), "fundamental_inliers": inliers,
        "fundamental_inlier_ratio": ratio,
        "orb_diagnostic_pass": inliers >= minimum and ratio >= .25,
    }


def main():
    args = arguments(); args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in args.images.iterdir()
                   if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    poses = load_poses(args.ground_truth)
    end = len(paths) - 1 if args.query_end < 0 else args.query_end
    pairs = select_pairs(poses, args.query_start, end,
                         args.min_frame_separation, args.max_distance_m,
                         args.max_rotation_deg, args.pairs)
    with (args.output / "selected_pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(
            ["query_frame", "candidate_frame", "distance_m", "rotation_deg"])
        writer.writerows(pairs)
    examples = args.output / "examples"; examples.mkdir(exist_ok=True)
    rclpy.init(args=["--ros-args", "--params-file", str(args.params)])
    node = HuMemSLAMNode(); rows = []
    try:
        for pair_index, (query_id, candidate_id, distance, angle) in enumerate(pairs):
            candidate_image = cv2.imread(str(paths[candidate_id]))
            natural_query = cv2.imread(str(paths[query_id]))
            candidate, _ = semantic_record(node, candidate_image, pair_index * 1000)
            pair_dir = examples / f"q{query_id:06d}_c{candidate_id:06d}"
            pair_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(pair_dir / "candidate_day.png"), candidate_image)
            cv2.imwrite(str(pair_dir / "query_day.png"), natural_query)
            for profile_index, name in enumerate(args.profiles):
                severity = profile_index / max(1, len(args.profiles) - 1)
                query_image = make_night(
                    natural_query, PROFILES[name],
                    args.seed + query_id * 100 + profile_index)
                query_image = add_nonuniform_lighting(
                    query_image, severity,
                    args.seed + query_id * 1000 + profile_index)
                query, latency = semantic_record(
                    node, query_image, pair_index * 1000 + profile_index + 1)
                breakdown = node.matcher.score_breakdown(
                    query, candidate, [query.scene], [candidate.scene])
                score = float(breakdown["unified_score"])
                rows.append({
                    "pair": pair_index + 1, "query_frame": query_id,
                    "candidate_frame": candidate_id, "distance_m": distance,
                    "rotation_deg": angle, "profile": name,
                    "human_score": score,
                    "human_pass": score > args.human_threshold,
                    "scene_score": breakdown["scene_score"],
                    "object_score": breakdown["object_score"],
                    "text_score": breakdown["text_score"],
                    "query_objects": len(query.static_objects),
                    "candidate_objects": len(candidate.static_objects),
                    "human_latency_ms": latency,
                    **orb_viewpoint_metrics(candidate_image, query_image,
                                            args.orb_min_inliers),
                })
                cv2.imwrite(str(pair_dir / f"query_{name}.png"), query_image)
        with (args.output / "paired_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));
            writer.writeheader(); writer.writerows(rows)
        summaries = []
        for name in args.profiles:
            subset = [row for row in rows if row["profile"] == name]
            summaries.append({
                "profile": name, "pairs": len(subset),
                "human_score_mean": float(np.mean([r["human_score"] for r in subset])),
                "human_pass_rate": float(np.mean([r["human_pass"] for r in subset])),
                "orb_inliers_mean": float(np.mean([r["fundamental_inliers"] for r in subset])),
                "orb_pass_rate": float(np.mean([r["orb_diagnostic_pass"] for r in subset])),
                "fast12_retention_mean": float(np.mean([r["fast12_retention"] for r in subset])),
            })
        with (args.output / "profile_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]));
            writer.writeheader(); writer.writerows(summaries)
        (args.output / "summary.json").write_text(json.dumps({
            "profiles": summaries,
            "warning": "ORB result is an OpenCV FAST/ORB/fundamental-RANSAC screening proxy, not full ORB-SLAM3.",
        }, indent=2) + "\n")
        x = np.arange(len(summaries)); fig, left = plt.subplots(figsize=(10, 5.8))
        left.plot(x, [s["human_score_mean"] for s in summaries], marker="o",
                  label="HuMemSLAM score"); left.axhline(args.human_threshold,
                  linestyle="--", label="Human threshold"); left.set_ylim(0, 1)
        left.set_ylabel("HuMemSLAM score"); right = left.twinx()
        right.plot(x, [s["orb_inliers_mean"] for s in summaries], marker="s",
                   color="tab:red", label="ORB/F inliers")
        right.axhline(args.orb_min_inliers, color="tab:red", linestyle="--",
                      label="ORB diagnostic boundary"); right.set_ylabel("RANSAC inliers")
        left.set_xticks(x, [s["profile"] for s in summaries]); left.grid(alpha=.2)
        lines, labels = left.get_legend_handles_labels(); lines2, labels2 = right.get_legend_handles_labels()
        left.legend(lines + lines2, labels + labels2); fig.tight_layout()
        fig.savefig(args.output / "humanslam_vs_orb_revisits.png", dpi=180); plt.close(fig)
        print(json.dumps(summaries, indent=2))
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
