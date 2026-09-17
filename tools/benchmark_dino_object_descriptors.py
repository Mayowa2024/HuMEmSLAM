#!/usr/bin/env python3
"""Benchmark dense DINOv2 YOLO-mask pooling on labelled revisit triplets."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dense_object_descriptor import (
    DenseDescriptorSpec, TensorRTDenseDescriptor, pool_mask_features,
)

MODES = ("centre_3x3", "centre_5x5", "mask", "mask_central")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--detector", type=Path,
                   default=ROOT / "weights/humanSLAM_YOLO_seg.engine")
    p.add_argument("--dino-config", type=Path,
                   default=ROOT / "weights/dinov2/dinov2_vits14_dense_448.json")
    p.add_argument("--confidence", type=float, default=0.5)
    return p.parse_args()


def process(detector, dino, image, confidence):
    detect_started = time.perf_counter()
    result = detector.predict(image, conf=confidence, device=0, verbose=False)[0]
    torch.cuda.synchronize()
    detector_ms = (time.perf_counter() - detect_started) * 1000.0
    if result.boxes is None or result.masks is None or not len(result.boxes):
        return [], detector_ms, 0.0, {mode: 0.0 for mode in MODES}
    dino_started = time.perf_counter()
    features = dino.describe(image)
    dino_ms = (time.perf_counter() - dino_started) * 1000.0
    height, width = image.shape[:2]
    objects = []
    pooling = {mode: 0.0 for mode in MODES}
    for index, box in enumerate(result.boxes):
        cls = int(box.cls.item())
        mask = result.masks.data[index].detach().float().cpu().numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        obj = {"class": result.names[cls], "confidence": float(box.conf.item())}
        for mode in MODES:
            started = time.perf_counter()
            obj[mode] = pool_mask_features(features, mask, mode)
            pooling[mode] += (time.perf_counter() - started) * 1000.0
        objects.append(obj)
    return objects, detector_ms, dino_ms, pooling


def similarity(query, candidate, mode):
    denominator = sum(obj["confidence"] for obj in query)
    if denominator <= 0:
        return 0.0
    total = 0.0
    for class_name in {obj["class"] for obj in query}:
        q = [obj for obj in query if obj["class"] == class_name and obj[mode] is not None]
        c = [obj for obj in candidate if obj["class"] == class_name and obj[mode] is not None]
        if not q or not c:
            continue
        matrix = np.asarray([[float(np.dot(a[mode], b[mode])) for b in c] for a in q])
        rows, cols = linear_sum_assignment(-matrix)
        total += sum(q[i]["confidence"] * float(np.clip((matrix[i, j]+1)/2, 0, 1))
                     for i, j in zip(rows, cols))
    return total / denominator


def main():
    a = arguments(); a.output.mkdir(parents=True, exist_ok=True)
    triplets = list(csv.DictReader(a.manifest.open(newline="", encoding="utf-8")))
    detector = YOLO(str(a.detector), task="segment")
    dino = TensorRTDenseDescriptor(DenseDescriptorSpec.from_json(a.dino_config))
    sample = cv2.imread(triplets[0]["query_path"])
    for _ in range(3):
        detector.predict(sample, conf=a.confidence, device=0, verbose=False)
        dino.describe(sample)
    cache, latency = {}, {"detector": [], "dino": [], **{m: [] for m in MODES}}

    def load(path):
        if path not in cache:
            image = cv2.imread(path)
            if image is None: raise FileNotFoundError(path)
            objects, detector_ms, dino_ms, pool_ms = process(
                detector, dino, image, a.confidence
            )
            cache[path] = objects
            latency["detector"].append(detector_ms)
            if dino_ms: latency["dino"].append(dino_ms)
            for mode in MODES: latency[mode].append(pool_ms[mode])
        return cache[path]

    rows = []
    for index, triplet in enumerate(triplets, 1):
        q, p, n = load(triplet["query_path"]), load(triplet["positive_path"]), load(triplet["negative_path"])
        row = {"query_frame": triplet["query_frame"], "query_objects": len(q),
               "positive_objects": len(p), "negative_objects": len(n)}
        for mode in MODES:
            ps, ns = similarity(q, p, mode), similarity(q, n, mode)
            row.update({f"{mode}_positive": ps, f"{mode}_negative": ns,
                        f"{mode}_margin": ps-ns, f"{mode}_correct": int(ps>ns)})
        rows.append(row)
        print(index, row["query_frame"], {m: round(row[f"{m}_margin"],3) for m in MODES})
    with (a.output/"pair_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    informative=[row for row in rows if row["query_objects"]]
    summary={"triplets":len(rows),"informative":len(informative),"modes":{},"latency":{}}
    for mode in MODES:
        summary["modes"][mode]={
            "strict_accuracy":float(np.mean([r[f"{mode}_correct"] for r in rows])),
            "informative_accuracy":float(np.mean([r[f"{mode}_correct"] for r in informative])) if informative else 0,
            "mean_margin":float(np.mean([r[f"{mode}_margin"] for r in rows])),
        }
    for name,values in latency.items():
        summary["latency"][name]={"mean_ms":float(np.mean(values)) if values else 0,
                                  "median_ms":float(np.median(values)) if values else 0,
                                  "p95_ms":float(np.percentile(values,95)) if values else 0}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
