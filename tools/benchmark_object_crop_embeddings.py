#!/usr/bin/env python3
"""Compare EigenPlaces and YOLO embeddings on class-matched object crops.

The benchmark uses pose-labelled query/positive/hard-negative image triplets.
YOLO supplies detections and bounding-box crops.  Each method embeds the same
crops; matching is restricted to equal semantic classes and uses one-to-one
Hungarian assignment.  The score denominator is determined by query objects,
matching HuMemSLAM's missing-evidence policy.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_place_descriptor import DescriptorSpec, TensorRTGlobalDescriptor


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--detector", type=Path, default=ROOT / "weights/humanSLAM_YOLO_seg.engine")
    p.add_argument("--yolo-features", type=Path, default=ROOT / "weights/humanSLAM_YOLO_seg.pt")
    p.add_argument("--eigenplaces", type=Path, default=ROOT / "weights/global_descriptors/eigenplaces_r18_512.json")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--max-objects", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def unit(v):
    v = np.asarray(v, np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def detect(detector, image, confidence, max_objects):
    t0 = time.perf_counter()
    result = detector.predict(image, conf=confidence, device=0, verbose=False)[0]
    torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000.0
    found = []
    if result.boxes is None:
        return found, latency
    h, w = image.shape[:2]
    boxes = sorted(result.boxes, key=lambda b: float(b.conf.item()), reverse=True)
    for box in boxes[:max_objects]:
        x1, y1, x2, y2 = box.xyxy[0].round().int().cpu().tolist()
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        cls = int(box.cls.item())
        found.append({
            "class_id": cls,
            "class_name": result.names[cls],
            "confidence": float(box.conf.item()),
            "crop": image[y1:y2, x1:x2].copy(),
        })
    return found, latency


def embed_eigen(model, objects):
    times = []
    for obj in objects:
        t0 = time.perf_counter()
        obj["eigenplaces"] = model.describe(obj["crop"])
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def embed_yolo(model, objects):
    if not objects:
        return []
    times = []


    for obj in objects:
        t0 = time.perf_counter()
        vectors = model.embed(obj["crop"], device=0, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        obj["yolo"] = unit(vectors[0].detach().float().cpu().numpy())
    return times


def set_similarity(query, candidate, key):
    """Confidence-weighted, class-specific, one-to-one crop similarity."""
    denominator = sum(obj["confidence"] for obj in query)
    if denominator <= 0:
        return 0.0, 0
    total, matches = 0.0, 0
    classes = sorted({obj["class_id"] for obj in query})
    for cls in classes:
        q = [o for o in query if o["class_id"] == cls]
        c = [o for o in candidate if o["class_id"] == cls]
        if not c:
            continue
        sims = np.asarray([[float(np.dot(a[key], b[key])) for b in c] for a in q])
        rows, cols = linear_sum_assignment(-sims)
        for i, j in zip(rows, cols):

            total += q[i]["confidence"] * float(np.clip((sims[i, j] + 1.0) / 2.0, 0, 1))
            matches += 1
    return total / denominator, matches


def main():
    a = args()
    a.output.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(newline="", encoding="utf-8") as f:
        triplets = list(csv.DictReader(f))
    detector = YOLO(str(a.detector), task="segment")
    yolo = YOLO(str(a.yolo_features), task="segment")
    eigen = TensorRTGlobalDescriptor(DescriptorSpec.from_json(a.eigenplaces))

    sample = cv2.imread(triplets[0]["query_path"])
    for _ in range(a.warmup):
        detector.predict(sample, conf=a.confidence, device=0, verbose=False)
        yolo.embed(sample, device=0, verbose=False)
        eigen.describe(sample)
    torch.cuda.synchronize()

    cache = {}
    latency = defaultdict(list)

    def process(path):
        path = str(path)
        if path in cache:
            return cache[path]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        objects, detect_ms = detect(detector, image, a.confidence, a.max_objects)
        e_times = embed_eigen(eigen, objects)
        y_times = embed_yolo(yolo, objects)
        latency["detector_image_ms"].append(detect_ms)
        latency["eigenplaces_crop_ms"].extend(e_times)
        latency["yolo_crop_ms"].extend(y_times)
        latency["eigenplaces_image_extra_ms"].append(sum(e_times))
        latency["yolo_image_extra_ms"].append(sum(y_times))
        cache[path] = objects
        return objects

    rows = []
    for index, triplet in enumerate(triplets, 1):
        q = process(triplet["query_path"])
        p = process(triplet["positive_path"])
        n = process(triplet["negative_path"])
        row = {
            "query_frame": triplet["query_frame"],
            "positive_frame": triplet["positive_frame"],
            "negative_frame": triplet["negative_frame"],
            "query_objects": len(q), "positive_objects": len(p), "negative_objects": len(n),
            "query_classes": json.dumps([o["class_name"] for o in q]),
        }
        for key in ("eigenplaces", "yolo"):
            ps, pm = set_similarity(q, p, key)
            ns, nm = set_similarity(q, n, key)
            row.update({f"{key}_positive": ps, f"{key}_negative": ns,
                        f"{key}_margin": ps - ns,
                        f"{key}_correct": int(ps > ns),
                        f"{key}_positive_matches": pm,
                        f"{key}_negative_matches": nm})
        rows.append(row)
        print(f"{index}/{len(triplets)} q={row['query_frame']} objects={len(q)} "
              f"Eigen margin={row['eigenplaces_margin']:+.3f} "
              f"YOLO margin={row['yolo_margin']:+.3f}")

    with (a.output / "pair_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"triplets": len(rows), "confidence": a.confidence,
               "informative_queries": sum(bool(r["query_objects"]) for r in rows)}
    for key in ("eigenplaces", "yolo"):
        informative = [r for r in rows if r["query_objects"]]
        summary[key] = {
            "strict_accuracy": float(np.mean([r[f"{key}_correct"] for r in rows])),
            "informative_accuracy": float(np.mean([r[f"{key}_correct"] for r in informative])) if informative else 0.0,
            "mean_margin": float(np.mean([r[f"{key}_margin"] for r in rows])),
        }
    summary["latency"] = {name: {
        "count": len(values), "mean_ms": float(np.mean(values)) if values else 0.0,
        "median_ms": float(np.median(values)) if values else 0.0,
        "p95_ms": float(np.percentile(values, 95)) if values else 0.0,
    } for name, values in latency.items()}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
