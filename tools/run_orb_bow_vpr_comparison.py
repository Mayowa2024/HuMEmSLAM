#!/usr/bin/env python3
"""Evaluate native ORB-SLAM3 ORB extraction and DBoW2 on fixed VPR manifests."""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from run_external_vpr_comparison import dataset_definition

ROOT = Path(__file__).resolve().parents[1]
ORB = Path("/home/teleopbike/ORB_SLAM3")


def full_precision(rows):
    wrong = [r["top1_score"] for r in rows if not r["top1_correct"]]
    threshold = max(wrong) + 1e-12 if wrong else -math.inf
    return sum(r["top1_correct"] and r["top1_score"] >= threshold for r in rows) / len(rows), threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--binary", type=Path, default=ROOT / "external/vpr_comparison/orb_bow_fixed_benchmark")
    args = parser.parse_args()
    data = dataset_definition(args.dataset, args.max_queries)
    output = args.output_root / args.dataset / "orb_bow"
    output.mkdir(parents=True, exist_ok=True)
    db_manifest, query_manifest = output / "database.tsv", output / "queries.tsv"
    db_manifest.write_text("".join(f"{f}\t{data['database_paths'][f]}\n" for f in data["database_frames"]))
    query_manifest.write_text("".join(f"{f}\t{data['query_paths'][f]}\n" for f in data["query_frames"]))
    raw = output / "native_rankings.csv"
    latency_path = output / "latency.csv"
    subprocess.run([
        str(args.binary), str(ORB / "Vocabulary/ORBvoc.txt"), str(db_manifest),
        str(query_manifest), str(raw), str(latency_path), str(args.latency_repetitions),
    ], check=True)

    rows = []
    for source in csv.DictReader(raw.open()):
        query = int(source["query_frame"])
        ranked = [
            int(source[f"rank{i}_frame"]) for i in range(1, 6)
            if source[f"rank{i}_frame"] != ""
        ]
        labels = [data["positive"](query, candidate) for candidate in ranked]
        first = labels.index(True) + 1 if any(labels) else None
        rows.append({
            "query_frame": query, "rank1_frame": ranked[0],
            "top1_score": float(source["rank1_score"]), "top1_correct": int(labels[0]),
            "first_correct_rank": first or "", "recall_at_1": int(labels[0]),
            "recall_at_5": int(any(labels)),
        })
    with (output / "query_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    latency = list(csv.DictReader(latency_path.open()))
    desc = np.asarray([float(r["descriptor_ms"]) for r in latency])
    end = np.asarray([float(r["end_to_end_ms"]) for r in latency])
    ranks = [int(r["first_correct_rank"]) for r in rows if r["first_correct_rank"] != ""]
    safe, threshold = full_precision(rows)
    summary = {
        "method": "orb_bow", "implementation": "native ORB-SLAM3 ORBextractor + DBoW2 ORBvoc",
        "dataset": args.dataset, "sequence": data["sequence"],
        "database_frames": len(data["database_frames"]), "evaluated_queries": len(rows),
        "recall@1": float(np.mean([r["recall_at_1"] for r in rows])),
        "recall@5": float(np.mean([r["recall_at_5"] for r in rows])),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(rows)),
        "empirical_recall_at_100_percent_precision": safe,
        "empirical_100_percent_precision_threshold": threshold,
        "latency_repetitions": args.latency_repetitions,
        "latency_queries_per_repetition": min(30, len(data["query_frames"])),
        "descriptor_latency_mean_ms": float(desc.mean()), "descriptor_latency_median_ms": float(np.median(desc)),
        "descriptor_latency_p95_ms": float(np.percentile(desc, 95)),
        "end_to_end_latency_mean_ms": float(end.mean()), "end_to_end_latency_median_ms": float(np.median(end)),
        "end_to_end_latency_p95_ms": float(np.percentile(end, 95)),
        "latency_mean_ms": float(end.mean()), "latency_median_ms": float(np.median(end)),
        "latency_p95_ms": float(np.percentile(end, 95)),
        "latency_scope": "decoded image -> grayscale/native ORB extraction -> vocabulary transform -> DBoW2 score -> top-5",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
