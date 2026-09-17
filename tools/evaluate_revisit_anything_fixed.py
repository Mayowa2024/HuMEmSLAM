#!/usr/bin/env python3
"""Score official Revisit Anything segment descriptors on the shared protocol."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import faiss
import numpy as np

from run_external_vpr_comparison import dataset_definition

ROOT = Path(__file__).resolve().parents[1]
REVISIT = ROOT / "external/vpr_comparison/Revisit-Anything"


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def safe_recall(rows):
    wrong = [row["top1_score"] for row in rows if not row["top1_correct"]]
    threshold = max(wrong) + 1e-12 if wrong else -math.inf
    value = sum(row["top1_correct"] and row["top1_score"] >= threshold for row in rows)
    return value / len(rows), threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()
    data = dataset_definition(args.dataset, args.max_queries)
    source = REVISIT / "workdir_data" / args.dataset / "out/fixed_manifest_bundle.pkl"
    bundle = pickle.loads(source.read_bytes())
    reference = normalize(bundle["reference_descriptors"].numpy())
    queries = normalize(bundle["query_descriptors"].numpy())
    reference_owner = np.asarray(bundle["reference_image_indices"], dtype=int)
    query_owner = np.asarray(bundle["query_image_indices"], dtype=int)

    raw = []
    for query_index, query_frame in enumerate(data["query_frames"]):
        allowed_images = np.asarray([
            index for index, frame in enumerate(data["database_frames"])
            if frame <= query_frame - 100
        ], dtype=int)
        allowed_segments = np.flatnonzero(np.isin(reference_owner, allowed_images))
        query_segments = np.flatnonzero(query_owner == query_index)
        index = faiss.IndexFlatL2(reference.shape[1])
        index.add(reference[allowed_segments])
        count = min(50, len(allowed_segments))
        distances, local_matches = index.search(queries[query_segments], count)
        matched_segments = allowed_segments[local_matches]
        raw.append((query_frame, distances, reference_owner[matched_segments]))

    all_distances = np.concatenate([item[1].reshape(-1) for item in raw])
    minimum, maximum = float(all_distances.min()), float(all_distances.max())
    rows = []
    for query_frame, distances, owners in raw:
        similarities = 2.0 - distances
        similarities = (similarities - (2.0 - maximum)) / max(maximum - minimum, 1e-12)
        scores = {}
        for segment_owners, segment_scores in zip(owners, similarities):
            for owner, score in zip(segment_owners, segment_scores):
                scores[int(owner)] = scores.get(int(owner), 0.0) + float(score)
        ranking = sorted(scores, key=scores.get, reverse=True)[:5]
        ranked_frames = [data["database_frames"][index] for index in ranking]
        labels = [data["positive"](query_frame, frame) for frame in ranked_frames]
        first = labels.index(True) + 1 if any(labels) else None
        rows.append({
            "query_frame": query_frame, "rank1_frame": ranked_frames[0],
            "top1_score": scores[ranking[0]] / max(len(distances), 1),
            "top1_correct": int(labels[0]), "first_correct_rank": first or "",
            "recall_at_1": int(labels[0]), "recall_at_5": int(any(labels)),
        })
    ranks = [int(row["first_correct_rank"]) for row in rows if row["first_correct_rank"] != ""]
    safe, threshold = safe_recall(rows)
    output = args.output_root / args.dataset / "revisit_anything"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "query_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "method": "revisit_anything",
        "implementation": "accelerated Revisit Anything: EfficientViT-SAM-L2, DINOv2 ViT-B/14, per-dataset reference VLAD-32, PCA-1024",
        "dataset": args.dataset, "sequence": data["sequence"],
        "database_frames": len(data["database_frames"]), "evaluated_queries": len(rows),
        "recall@1": float(np.mean([r["recall_at_1"] for r in rows])),
        "recall@5": float(np.mean([r["recall_at_5"] for r in rows])),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(rows)),
        "empirical_recall_at_100_percent_precision": safe,
        "empirical_100_percent_precision_threshold": threshold,
        "latency_status": "pending standardised three-pass timing",
    }
    (output / "accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
