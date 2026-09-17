#!/usr/bin/env python3
"""Calibrate semantic submission thresholds on a frozen 4Seasons ranking."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from tools.evaluate_4seasons_retrieval import reference_poses, quaternion_angle_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-threshold", type=float, default=0.70)
    parser.add_argument("--max-proposals-per-query", type=float, default=3.10)
    parser.add_argument("--max-precision-drop", type=float, default=0.015)
    args = parser.parse_args()

    positions, quaternions = reference_poses(args.sequence)
    queries = defaultdict(list)
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["rank"]) <= 5:
                queries[int(row["query_frame_id"])].append((
                    float(row["score"]), int(row["candidate_source_frame_id"])
                ))

    def correct(query, candidate):
        return (
            query - candidate >= 100
            and np.linalg.norm(positions[query] - positions[candidate]) <= 5.0
            and quaternion_angle_deg(
                quaternions[query], quaternions[candidate]
            ) <= 30.0
        )

    def measure(threshold, validation):
        true_positive = false_positive = hits = possible = proposals = count = 0
        for query, candidates in queries.items():
            if (query % 5 < 3) != validation:
                continue
            count += 1
            available = [correct(query, candidate)
                         for _, candidate in candidates]
            selected = [correct(query, candidate)
                        for score, candidate in candidates if score > threshold]
            true_positive += sum(selected)
            false_positive += len(selected) - sum(selected)
            proposals += len(selected)
            if any(available):
                possible += 1
                hits += int(any(selected))
        return {
            "threshold": float(threshold),
            "query_recall": float(hits / possible if possible else 0.0),
            "proposal_precision": (
                float(true_positive / (true_positive + false_positive))
                if true_positive + false_positive else 1.0
            ),
            "proposals_per_query": float(proposals / count if count else 0.0),
            "correct_proposals": int(true_positive),
            "queries": int(count),
        }

    baseline = measure(args.baseline_threshold, True)
    sweep = [measure(float(value), True)
             for value in np.arange(0.30, 0.801, 0.005)]
    eligible = [row for row in sweep if (
        row["proposals_per_query"] <= args.max_proposals_per_query
        and row["proposal_precision"]
        >= baseline["proposal_precision"] - args.max_precision_drop
    )]
    selected = max(eligible, key=lambda row: (
        row["query_recall"], row["proposal_precision"], row["threshold"]
    ))
    held_out = measure(selected["threshold"], False)
    baseline_held_out = measure(args.baseline_threshold, False)
    result = {
        "selection_split": "query_frame_id % 5 < 3",
        "held_out_split": "query_frame_id % 5 >= 3",
        "objective": "maximise correct-query proposal recall subject to proposal budget and bounded precision loss",
        "constraints": {
            "max_proposals_per_query": args.max_proposals_per_query,
            "max_precision_drop": args.max_precision_drop,
        },
        "validation_baseline": baseline,
        "validation_selected": selected,
        "held_out_baseline": baseline_held_out,
        "held_out_selected": held_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
