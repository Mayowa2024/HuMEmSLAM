#!/usr/bin/env python3
"""Fixed-database native-PyTorch VPR comparison for dissertation datasets.

This runner intentionally does not use TensorRT or ONNX.  It currently
supports the official PyTorch SALAD and MegaLoc releases.  Database/query
manifests are deterministic and shared by every method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CARLA_DATA_ROOT = Path(
    "/media/teleopbike/Windows/Users/teleo/OneDrive - Oxford Brookes University/"
    "Natalie Mouradian's files - carla_slam_data_retry"
)
CARLA_GT_ROOT = (
    ROOT / "test_results/dissertation_final/carla_retry/"
    "paired_weather_3x_20260822/ground_truth"
)
CARLA_DATASETS = {
    "carla_extreme_rain_fog": "pair_01_extreme_rain_fog",
    "carla_deep_night": "pair_02_deep_night",
    "carla_dense_fog_overcast": "pair_03_dense_fog_overcast",
    "carla_extreme_sunset_glare": "pair_04_extreme_sunset_glare",
    "carla_clear_noon_repeat": "pair_05_clear_noon_repeat",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("salad", "megaloc"), required=True)
    parser.add_argument("--dataset", choices=(
        "kitti06_clean", "kitti06_b15_d80", "kitti06_b35_d80",
        "business_fall", "garage_feb2021",
        *CARLA_DATASETS,
    ), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir()
         if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: int("".join(c for c in path.stem if c.isdigit()) or -1),
    )


def kitti_poses(path: Path):
    rows = np.loadtxt(path).reshape(-1, 3, 4)
    positions = rows[:, :, 3]
    rotations = rows[:, :, :3]
    return positions, rotations


def fourseasons_poses(sequence: Path):
    camera_times = []
    for line in (sequence / "times.txt").read_text().splitlines():
        values = line.split()
        if values:
            camera_times.append(float(values[1]) if len(values) > 1 else float(values[0]))
    times = np.asarray(camera_times)
    reference = np.loadtxt(sequence / "result.txt")
    indices = np.searchsorted(reference[:, 0], times)
    indices = np.clip(indices, 1, len(reference) - 1)
    before = indices - 1
    use_before = abs(reference[before, 0] - times) <= abs(reference[indices, 0] - times)
    indices[use_before] = before[use_before]
    return reference[indices, 1:4], reference[indices, 4:8]


def rotation_difference_deg(first, second, quaternion=False):
    if quaternion:
        first = first / max(float(np.linalg.norm(first)), 1e-12)
        second = second / max(float(np.linalg.norm(second)), 1e-12)
        dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
        return math.degrees(2.0 * math.acos(dot))
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evenly_select(values, count):
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[index] for index in sorted(set(indices))]


def dataset_definition(name: str, max_queries: int):
    kitti_clean = Path(
        "/home/teleopbike/Documents/slam_experiments/datasets/"
        "KITTI/dataset/sequences/06/image_0"
    )
    kitti_gt = Path(
        "/home/teleopbike/Documents/slam_experiments/datasets/"
        "KITTI/dataset/poses/06.txt"
    )
    if name.startswith("kitti06_"):
        query_roots = {
            "kitti06_clean": kitti_clean,
            "kitti06_b15_d80": ROOT / "test_scenarios/final_kitti06/K06-4_blur15_dark80/image_0",
            "kitti06_b35_d80": ROOT / "test_scenarios/final_kitti06/K06-5_blur35_dark80/image_0",
        }
        positions, orientations = kitti_poses(kitti_gt)
        database_paths = image_paths(kitti_clean)
        query_paths = image_paths(query_roots[name])
        database_frames = list(range(0, min(828, len(database_paths)), 5))
        quaternion = False
        candidate_queries = range(828, min(len(query_paths), len(positions)))
        sequence = "KITTI 06"
    elif name in CARLA_DATASETS:
        folder = CARLA_DATASETS[name]
        sequence_root = CARLA_DATA_ROOT / folder
        database_paths = image_paths(sequence_root / "image_0")
        query_paths = database_paths
        positions, orientations = kitti_poses(CARLA_GT_ROOT / f"{folder}.txt")
        end = min(len(database_paths), len(positions))




        transition_frame = 1520
        database_frames = list(range(0, min(transition_frame, end), 5))
        candidate_queries = range(transition_frame, end)
        quaternion = False
        sequence = f"CARLA Town10HD {folder}"
    else:
        sequence_roots = {
            "business_fall": ROOT / (
                "test_scenarios/4seasons_business_school/"
                "recording_2020-10-08_09-30-57"
            ),
            "garage_feb2021": ROOT / (
                "test_scenarios/4seasons_multilevel_carpark/"
                "recording_2021-02-25_13-39-06"
            ),
        }
        sequence_root = sequence_roots[name]
        database_paths = image_paths(sequence_root / "undistorted_images/cam0")
        query_paths = database_paths
        positions, orientations = fourseasons_poses(sequence_root)
        end = min(len(database_paths), len(positions))
        if name == "business_fall":

            end = min(end, 10742)
        stride = 20 if name == "business_fall" else 10
        database_frames = list(range(0, end, stride))
        quaternion = True
        candidate_queries = range(100, end)
        sequence = sequence_root.name

    def positive(query, candidate):
        if query - candidate < 100:
            return False
        if np.linalg.norm(positions[query] - positions[candidate]) > 5.0:
            return False
        return rotation_difference_deg(
            orientations[query], orientations[candidate], quaternion
        ) <= 30.0

    eligible = [
        query for query in candidate_queries
        if any(positive(query, candidate) for candidate in database_frames
               if candidate <= query - 100)
    ]
    query_frames = evenly_select(eligible, max_queries)
    return {
        "name": name,
        "sequence": sequence,
        "database_paths": database_paths,
        "query_paths": query_paths,
        "database_frames": database_frames,
        "query_frames": query_frames,
        "frame_count": min(len(query_paths), len(positions)),
        "positions": positions,
        "orientations": orientations,
        "quaternion": quaternion,
        "positive": positive,
    }


def load_model(name: str, device: torch.device):
    if name == "salad":
        repo = Path.home() / ".cache/torch/hub/serizba_salad_main"
        model = torch.hub.load(
            str(repo), "dinov2_salad", source="local", pretrained=True
        )
    else:
        repo = ROOT / "external/vpr_comparison/MegaLoc"
        model = torch.hub.load(
            str(repo), "get_trained_model", source="local"
        )
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize(size=[322, 322], antialias=True),
    ])
    return model, transform


def describe(model, transform, path: Path, device: torch.device):
    source = Image.open(path).convert("RGB")
    image = transform(source).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        descriptor = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = (time.perf_counter() - started) * 1000.0
    descriptor = descriptor.detach().float().cpu().numpy().reshape(-1)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor, latency


def timed_descriptor(model, transform, source, device):
    """Time preprocessing, descriptor inference and L2 normalisation, not I/O."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    image = transform(source).unsqueeze(0).to(device)
    with torch.inference_mode():
        descriptor = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    descriptor = descriptor.detach().float().cpu().numpy().reshape(-1)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor, (time.perf_counter() - started) * 1000.0


def recall_at_full_precision(rows):
    wrong = [row["top1_score"] for row in rows if not row["top1_correct"]]
    threshold = max(wrong) + 1e-12 if wrong else -math.inf
    accepted_correct = sum(
        row["top1_correct"] and row["top1_score"] >= threshold for row in rows
    )
    return accepted_correct / len(rows), threshold


def main():
    args = arguments()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not record CPU latency as GPU latency")
    data = dataset_definition(args.dataset, args.max_queries)
    output = args.output_root / args.dataset / args.model
    output.mkdir(parents=True, exist_ok=True)
    model, transform = load_model(args.model, device)

    warm_path = data["database_paths"][data["database_frames"][0]]
    for _ in range(args.warmups):
        describe(model, transform, warm_path, device)

    required = sorted(set(data["database_frames"] + data["query_frames"]))
    descriptors = {}
    accuracy_latencies = []
    for frame in required:
        source = (
            data["query_paths"][frame]
            if frame in set(data["query_frames"])
            else data["database_paths"][frame]
        )
        descriptors[frame], latency = describe(model, transform, source, device)
        accuracy_latencies.append(latency)

    rows = []
    for query in data["query_frames"]:
        candidates = [
            frame for frame in data["database_frames"] if frame <= query - 100
        ]
        matrix = np.stack([descriptors[frame] for frame in candidates])
        scores = matrix @ descriptors[query]
        order = np.argsort(-scores)
        ranked = [candidates[index] for index in order[:5]]
        labels = [data["positive"](query, candidate) for candidate in ranked]
        first = labels.index(True) + 1 if any(labels) else None
        rows.append({
            "query_frame": query,
            "candidate_count": len(candidates),
            "rank1_frame": ranked[0],
            "rank1_distance_m": float(np.linalg.norm(
                data["positions"][query] - data["positions"][ranked[0]]
            )),
            "top1_score": float(scores[order[0]]),
            "top1_correct": int(labels[0]),
            "first_correct_rank": first or "",
            "recall_at_1": int(labels[0]),
            "recall_at_5": int(any(labels)),
        })

    latency_rows = []
    timed_queries = evenly_select(data["query_frames"], min(30, len(data["query_frames"])))
    for repetition in range(1, args.latency_repetitions + 1):
        for query in timed_queries:

            source = Image.open(data["query_paths"][query]).convert("RGB")
            candidates = [
                frame for frame in data["database_frames"] if frame <= query - 100
            ]
            matrix = np.stack([descriptors[frame] for frame in candidates])
            descriptor, descriptor_ms = timed_descriptor(
                model, transform, source, device
            )
            end_started = time.perf_counter()

            scores = matrix @ descriptor
            np.argsort(-scores)[:5]
            search_ms = (time.perf_counter() - end_started) * 1000.0
            latency_rows.append({
                "repetition": repetition,
                "query_frame": query,
                "descriptor_ms": descriptor_ms,
                "search_ms": search_ms,
                "end_to_end_ms": descriptor_ms + search_ms,
            })

    successful_ranks = [int(row["first_correct_rank"]) for row in rows
                        if row["first_correct_rank"] != ""]
    recall_full_precision, threshold = recall_at_full_precision(rows)
    descriptor_values = [row["descriptor_ms"] for row in latency_rows]
    latency_values = [row["end_to_end_ms"] for row in latency_rows]
    summary = {
        "method": args.model,
        "implementation": "official native PyTorch",
        "dataset": args.dataset,
        "sequence": data["sequence"],
        "database_frames": len(data["database_frames"]),
        "eligible_queries_before_sampling": len([
            q for q in range(data["frame_count"])
            if q >= 100 and any(data["positive"](q, c)
                                for c in data["database_frames"] if c <= q - 100)
        ]),
        "evaluated_queries": len(rows),
        "recall@1": float(np.mean([row["recall_at_1"] for row in rows])),
        "recall@5": float(np.mean([row["recall_at_5"] for row in rows])),
        "mrr": float(sum(1.0 / rank for rank in successful_ranks) / len(rows)),
        "empirical_recall_at_100_percent_precision": recall_full_precision,
        "empirical_100_percent_precision_threshold": threshold,
        "latency_repetitions": args.latency_repetitions,
        "latency_queries_per_repetition": len(timed_queries),
        "latency_mean_ms": float(np.mean(latency_values)),
        "latency_median_ms": float(np.median(latency_values)),
        "latency_p95_ms": float(np.percentile(latency_values, 95)),
        "descriptor_latency_mean_ms": float(np.mean(descriptor_values)),
        "descriptor_latency_median_ms": float(np.median(descriptor_values)),
        "descriptor_latency_p95_ms": float(np.percentile(descriptor_values, 95)),
        "end_to_end_latency_mean_ms": float(np.mean(latency_values)),
        "end_to_end_latency_median_ms": float(np.median(latency_values)),
        "end_to_end_latency_p95_ms": float(np.percentile(latency_values, 95)),
        "latency_scope": (
            "decoded image -> preprocessing -> descriptor -> database cosine "
            "similarity -> top-5 ranking; image I/O/database construction excluded"
        ),
        "device": str(device),
        "warning": (
            "Recall at 100% precision is an empirical same-set ceiling. "
            "Use a validation-calibrated threshold for the final test claim."
        ),
    }

    with (output / "query_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latency_rows[0]))
        writer.writeheader(); writer.writerows(latency_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "database_frames": data["database_frames"],
        "query_frames": data["query_frames"],
        "database_image_root": str(data["database_paths"][0].parent),
        "query_image_root": str(data["query_paths"][0].parent),
        "temporal_exclusion_frames": 100,
        "position_threshold_m": 5.0,
        "orientation_threshold_deg": 30.0,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
