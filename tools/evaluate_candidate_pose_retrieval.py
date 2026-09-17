#!/usr/bin/env python3
"""Evaluate HuMemSLAM candidate rankings against KITTI ground-truth position."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revisit-start", type=int, default=835)
    parser.add_argument("--min-frame-separation", type=int, default=300)
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    return parser.parse_args()


def load_positions(path):
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    return poses[:, :, 3]


def load_rankings(path, revisit_start):
    queries = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            query = int(row["query_frame_id"])
            if query < revisit_start:
                continue
            candidate_value = row.get("candidate_source_frame_id", "")
            if not candidate_value:
                continue
            queries[query].append((int(row["rank"]), int(candidate_value)))
    return {
        query: sorted(candidates)
        for query, candidates in queries.items()
    }


def evaluate(args):
    positions = load_positions(args.groundtruth)
    rankings = load_rankings(args.candidates, args.revisit_start)
    ks = (1, 5, 10, 25)
    successes = {k: 0 for k in ks}
    rows = []

    for query, candidates in sorted(rankings.items()):
        evaluated = []
        for rank, candidate in candidates:
            valid = query < len(positions) and candidate < len(positions)
            distance = (
                float(np.linalg.norm(positions[query] - positions[candidate]))
                if valid else float("inf")
            )
            is_prior_visit = query - candidate >= args.min_frame_separation
            correct = is_prior_visit and distance <= args.distance_threshold
            evaluated.append((rank, candidate, distance, is_prior_visit, correct))

        for k in ks:
            successes[k] += int(any(item[4] for item in evaluated if item[0] <= k))

        top = evaluated[0]
        correct_items = [item for item in evaluated if item[4]]
        rows.append({
            "query_frame_id": query,
            "top1_candidate_source_frame_id": top[1],
            "top1_pose_distance_m": top[2],
            "top1_is_prior_visit": int(top[3]),
            "top1_correct": int(top[4]),
            "first_correct_rank": correct_items[0][0] if correct_items else "",
            "first_correct_candidate_source_frame_id": (
                correct_items[0][1] if correct_items else ""
            ),
            "first_correct_pose_distance_m": (
                correct_items[0][2] if correct_items else ""
            ),
        })

    count = len(rows)
    summary = {
        "candidate_csv": str(args.candidates.resolve()),
        "groundtruth": str(args.groundtruth.resolve()),
        "revisit_start": args.revisit_start,
        "min_frame_separation": args.min_frame_separation,
        "distance_threshold_m": args.distance_threshold,
        "query_count": count,
        **{
            f"recall_at_{k}": successes[k] / count if count else 0.0
            for k in ks
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "retrieval_queries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    values = [summary[f"recall_at_{k}"] for k in ks]
    axis.plot(ks, values, marker="o", linewidth=2)
    axis.set_xticks(ks)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("K")
    axis.set_ylabel("Recall@K")
    axis.set_title("HuMemSLAM pose-ground-truth retrieval")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "retrieval_recall_at_k.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate(parse_args())
