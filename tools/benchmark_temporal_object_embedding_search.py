#!/usr/bin/env python3
"""Test cached temporal object embeddings when the current query has no detections."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_object_crop_embeddings import detect, embed_eigen
sys.path.insert(0, str(ROOT))
from global_place_descriptor import DescriptorSpec, TensorRTGlobalDescriptor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--decay", type=float, default=0.70)
    p.add_argument("--confidence", type=float, default=0.50)
    p.add_argument("--max-objects", type=int, default=8)
    p.add_argument("--detector", type=Path, default=ROOT / "weights/humanSLAM_YOLO_seg.engine")
    p.add_argument("--eigenplaces", type=Path, default=ROOT / "weights/global_descriptors/eigenplaces_r18_512.json")
    return p.parse_args()


def cached_query_similarity(history, candidate, decay):
    """Search each cached observation against same-class candidate objects."""
    weighted, denominator, comparisons = 0.0, 0.0, 0
    started = time.perf_counter()
    for obj, age in history:
        weight = obj["confidence"] * (decay ** age)
        denominator += weight
        peers = [c for c in candidate if c["class_id"] == obj["class_id"]]
        if not peers:
            continue
        similarities = [float(np.dot(obj["eigenplaces"], c["eigenplaces"])) for c in peers]
        weighted += weight * float(np.clip((max(similarities) + 1.0) / 2.0, 0, 1))
        comparisons += len(peers)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return (weighted / denominator if denominator else 0.0), elapsed_ms, comparisons


def main():
    a = parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    triplets = list(csv.DictReader(a.manifest.open(newline="", encoding="utf-8")))
    detector = YOLO(str(a.detector), task="segment")
    eigen = TensorRTGlobalDescriptor(DescriptorSpec.from_json(a.eigenplaces))
    sample = cv2.imread(triplets[0]["query_path"])
    detector.predict(sample, conf=a.confidence, device=0, verbose=False)
    eigen.describe(sample); torch.cuda.synchronize()
    cache = {}

    def process(path):
        key = str(path)
        if key in cache:
            return cache[key]
        image = cv2.imread(key)
        if image is None:
            raise FileNotFoundError(key)
        objects, detector_ms = detect(detector, image, a.confidence, a.max_objects)
        embedding_times = embed_eigen(eigen, objects)
        cache[key] = (objects, detector_ms, sum(embedding_times))
        return cache[key]

    database_paths = sorted({
        str(triplet[f"{role}_path"])
        for triplet in triplets for role in ("positive", "negative")
    })
    database = {path: process(path)[0] for path in database_paths}

    rows = []
    for i, triplet in enumerate(triplets, 1):
        query_path = Path(triplet["query_path"])
        frame = int(triplet["query_frame"])
        history, detector_update_ms, embedding_update_ms = [], 0.0, 0.0
        for age in range(1, a.history + 1):
            path = query_path.with_name(f"{frame-age:06d}{query_path.suffix}")
            objects, detector_ms, embedding_ms = process(path)
            detector_update_ms += detector_ms
            embedding_update_ms += embedding_ms
            history.extend((obj, age - 1) for obj in objects)
        positive, _, _ = process(triplet["positive_path"])
        negative, _, _ = process(triplet["negative_path"])
        ps, p_search, pc = cached_query_similarity(history, positive, a.decay)
        ns, n_search, nc = cached_query_similarity(history, negative, a.decay)
        global_started = time.perf_counter()
        global_scores = [
            (cached_query_similarity(history, objects, a.decay)[0], path)
            for path, objects in database.items()
        ]
        global_search_ms = (time.perf_counter() - global_started) * 1000.0
        global_scores.sort(reverse=True)
        positive_path = str(triplet["positive_path"])
        global_rank = next(
            rank for rank, (_, path) in enumerate(global_scores, 1)
            if path == positive_path
        )
        rows.append({
            "query_frame": frame,
            "positive_frame": triplet["positive_frame"],
            "negative_frame": triplet["negative_frame"],
            "cached_observations": len(history),
            "positive_score": ps, "negative_score": ns,
            "margin": ps - ns, "correct": int(ps > ns),
            "history_detector_ms": detector_update_ms,
            "history_embedding_ms": embedding_update_ms,
            "history_update_ms": detector_update_ms + embedding_update_ms,
            "search_ms": p_search + n_search,
            "vector_comparisons": pc + nc,
            "database_images": len(database),
            "global_positive_rank": global_rank,
            "global_recall_at_1": int(global_rank <= 1),
            "global_recall_at_5": int(global_rank <= 5),
            "global_search_ms": global_search_ms,
        })
        print(f"{i}/{len(triplets)} q={frame} cached={len(history)} margin={ps-ns:+.3f}")
    with (a.output / "pair_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    informative = [r for r in rows if r["cached_observations"]]
    summary = {
        "triplets": len(rows), "history_frames": a.history, "decay": a.decay,
        "informative": len(informative),
        "strict_accuracy": float(np.mean([r["correct"] for r in rows])),
        "informative_accuracy": float(np.mean([r["correct"] for r in informative])) if informative else 0.0,
        "mean_margin": float(np.mean([r["margin"] for r in rows])),
        "mean_history_update_ms_for_three_frames": float(np.mean([r["history_update_ms"] for r in rows])),
        "mean_detector_ms_for_three_frames": float(np.mean([r["history_detector_ms"] for r in rows])),
        "mean_added_embedding_ms_for_three_frames": float(np.mean([r["history_embedding_ms"] for r in rows])),
        "mean_added_embedding_ms_per_history_frame": float(np.mean([r["history_embedding_ms"] for r in rows])) / a.history,
        "mean_search_ms_for_positive_and_negative": float(np.mean([r["search_ms"] for r in rows])),
        "median_search_ms_for_positive_and_negative": float(np.median([r["search_ms"] for r in rows])),
        "database_images": len(database),
        "global_recall_at_1": float(np.mean([r["global_recall_at_1"] for r in rows])),
        "global_recall_at_5": float(np.mean([r["global_recall_at_5"] for r in rows])),
        "mean_global_search_ms": float(np.mean([r["global_search_ms"] for r in rows])),
        "p95_global_search_ms": float(np.percentile([r["global_search_ms"] for r in rows], 95)),
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
