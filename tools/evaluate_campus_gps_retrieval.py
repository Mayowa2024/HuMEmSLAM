#!/usr/bin/env python3
"""Evaluate ranked online retrieval using interpolated campus GNSS positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RADII = (2, 3, 4, 5, 6, 7, 8)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_keyframes(path):
    rows = []
    by_id = {}
    for row in read_rows(path):
        if row.get("gps_valid") != "1":
            continue
        item = {
            "id": int(row["keyframe_id"]),
            "frame": int(row["frame_id"]),
            "time": float(row["dataset_time"]),
            "x": float(row["east_m"]),
            "y": float(row["north_m"]),
        }
        rows.append(item)
        by_id[item["id"]] = item
    rows.sort(key=lambda item: item["time"])
    return rows, by_id


def details(text):
    result = {}
    for field in text.split(";"):
        if "=" in field:
            key, value = field.strip().split("=", 1)
            result[key] = value
    return result


def load_baseline_ranked(path):
    proposals = {}
    for row in read_rows(path):
        if row.get("event") != "RETRIEVAL_CANDIDATE":
            continue
        item = details(row.get("details", ""))
        if item.get("source") != "native_bow" or item.get("stage") != "raw_shortlist":
            continue
        try:
            query = int(item["query_kf"])
            rank = int(item["rank"])
            candidate = int(item["candidate_kf"])
        except (KeyError, ValueError):
            continue
        if 1 <= rank <= 5:
            proposals.setdefault(query, {}).setdefault(rank, candidate)
    return proposals


def load_humanslam_ranked(path):
    proposals = {}
    for row in read_rows(path):
        try:
            rank = int(row["rank"])
            query = int(row["query_reference_keyframe_id"])
            candidate = int(row["candidate_keyframe_id"])
        except (KeyError, ValueError):
            continue
        if 1 <= rank <= 5:
            proposals.setdefault(query, {}).setdefault(rank, candidate)
    return proposals


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def evaluate(keyframes, by_id, proposals, temporal_exclusion_s):
    results = []
    for radius in RADII:
        eligible = []
        for query_id in proposals:
            query = by_id.get(query_id)
            if query is None:
                continue
            if any(
                query["time"] - prior["time"] >= temporal_exclusion_s
                and dist(query, prior) <= radius
                for prior in keyframes
                if prior["time"] < query["time"]
            ):
                eligible.append(query_id)
        correct_at_1 = 0
        correct_at_5 = 0
        reciprocal_rank_sum = 0.0
        evaluable_at_1 = 0
        for query_id in eligible:
            query = by_id[query_id]
            ranked = proposals.get(query_id, {})
            rank1 = by_id.get(ranked.get(1))
            if rank1 is not None:
                evaluable_at_1 += 1
            first_correct_rank = None
            for rank in sorted(ranked):
                candidate = by_id.get(ranked[rank])
                if candidate is None:
                    continue
                if (query["time"] - candidate["time"] >= temporal_exclusion_s
                        and dist(query, candidate) <= radius):
                    first_correct_rank = rank
                    break
            if first_correct_rank == 1:
                correct_at_1 += 1
            if first_correct_rank is not None:
                correct_at_5 += 1
                reciprocal_rank_sum += 1.0 / first_correct_rank
        results.append({
            "radius_m": radius,
            "retrieval_queries": len(proposals),
            "eligible_queries": len(eligible),
            "gps_evaluable_rank1": evaluable_at_1,
            "correct_rank1": correct_at_1,
            "correct_rank5": correct_at_5,
            "recall_at_1": correct_at_1 / len(eligible) if eligible else None,
            "recall_at_5": correct_at_5 / len(eligible) if eligible else None,
            "mrr": reciprocal_rank_sum / len(eligible) if eligible else None,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-exclusion", type=float, default=30.0)
    parser.add_argument("--case", action="append", nargs=4,
                        metavar=("DATASET", "METHOD", "KEYFRAMES_GPS", "PROPOSALS"),
                        required=True)
    args = parser.parse_args()
    output = []
    for dataset, method, keyframes_path, proposal_path in args.case:
        keyframes, by_id = load_keyframes(keyframes_path)
        proposals = (load_baseline_ranked(proposal_path) if method == "baseline"
                     else load_humanslam_ranked(proposal_path))
        for row in evaluate(keyframes, by_id, proposals, args.temporal_exclusion):
            row.update({"dataset": dataset, "method": method,
                        "temporal_exclusion_s": args.temporal_exclusion})
            output.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "radius_m", "temporal_exclusion_s",
              "retrieval_queries", "eligible_queries", "gps_evaluable_rank1",
              "correct_rank1", "correct_rank5", "recall_at_1", "recall_at_5", "mrr"]
    with (args.output / "recall_by_radius.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (args.output / "recall_by_radius.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
