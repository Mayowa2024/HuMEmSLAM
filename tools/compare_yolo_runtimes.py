#!/usr/bin/env python3
"""Compare COCO PT, Mapillary PT and TensorRT on an identical image manifest."""

import argparse
import csv
import json
import gc
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "validation_results/yolo26n_seg_mapillary_domain_test/prediction_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "validation_results/yolo_model_tradeoff",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_stats(values):
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"dataset_name", "source_image_directory", "source_image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid or empty prediction manifest: {path}")
    return rows


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows = load_manifest(manifest)

    specifications = [
        {
            "name": "COCO_PRETRAINED_YOLO26N_SEG_PT",
            "path": root / "training_data/pretrained/yolo26n-seg.pt",
            "training_domain": "COCO",
            "runtime": "PyTorch",
        },
        {
            "name": "MAPILLARY_FINETUNED_HUMANSLAM_PT",
            "path": root / "weights/humanSLAM_YOLO_seg.pt",
            "training_domain": "Mapillary Vistas HuMemSLAM 21-class subset",
            "runtime": "PyTorch",
        },
        {
            "name": "MAPILLARY_FINETUNED_HUMANSLAM_TENSORRT_FP16",
            "path": root / "weights/humanSLAM_YOLO_seg.engine",
            "training_domain": "Mapillary Vistas HuMemSLAM 21-class subset",
            "runtime": "TensorRT FP16",
        },
    ]

    prediction_rows = []
    summaries = {}
    source_paths = [row["source_image"] for row in source_rows]

    for specification in specifications:
        model_path = specification["path"].resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = YOLO(str(model_path), task="segment")
        model.predict(
            source=source_paths[0],
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        model_output = output_root / specification["name"]
        class_counts = Counter()
        dataset_counts = defaultdict(Counter)
        pipeline_times = []
        inference_times = []
        wall_times = []
        images_with_detections = 0

        for source_row in source_rows:
            wall_started = time.perf_counter()
            result = model.predict(
                source=source_row["source_image"],
                conf=args.confidence,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            dataset_name = source_row["dataset_name"]
            dataset_output = model_output / dataset_name
            dataset_output.mkdir(parents=True, exist_ok=True)
            source_path = Path(source_row["source_image"])
            sample_index = int(source_row.get("sample_index", 0))
            output_overlay = (
                dataset_output
                / f"{sample_index:03d}_{source_path.stem}_prediction.jpg"
            )
            if not cv2.imwrite(str(output_overlay), result.plot()):
                raise RuntimeError(f"Failed to save overlay: {output_overlay}")

            detections = []
            if result.boxes is not None:
                class_ids = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for class_id, confidence in zip(class_ids, confidences):
                    class_name = result.names[int(class_id)]
                    class_counts[class_name] += 1
                    dataset_counts[dataset_name][class_name] += 1
                    detections.append(f"{class_name}:{float(confidence):.4f}")
            if detections:
                images_with_detections += 1

            preprocess_ms = float(result.speed.get("preprocess", 0.0))
            inference_ms = float(result.speed.get("inference", 0.0))
            postprocess_ms = float(result.speed.get("postprocess", 0.0))
            pipeline_ms = preprocess_ms + inference_ms + postprocess_ms
            pipeline_times.append(pipeline_ms)
            inference_times.append(inference_ms)
            wall_times.append(wall_ms)
            prediction_rows.append(
                {
                    "model_name": specification["name"],
                    "model_path": str(model_path),
                    "dataset_name": dataset_name,
                    "source_image_directory": source_row["source_image_directory"],
                    "source_image": str(source_path),
                    "output_overlay": str(output_overlay),
                    "detection_count": len(detections),
                    "mask_count": len(result.masks) if result.masks is not None else 0,
                    "preprocess_ms": f"{preprocess_ms:.6f}",
                    "inference_ms": f"{inference_ms:.6f}",
                    "postprocess_ms": f"{postprocess_ms:.6f}",
                    "pipeline_ms": f"{pipeline_ms:.6f}",
                    "iterator_wall_ms": f"{wall_ms:.6f}",
                    "detections": ";".join(detections),
                }
            )

        summaries[specification["name"]] = {
            "model_path": str(model_path),
            "training_domain": specification["training_domain"],
            "runtime": specification["runtime"],
            "class_count": len(model.names),
            "images": len(source_rows),
            "images_with_detections": images_with_detections,
            "total_detections": sum(class_counts.values()),
            "total_masks": sum(
                int(row["mask_count"])
                for row in prediction_rows
                if row["model_name"] == specification["name"]
            ),
            "pipeline_latency": latency_stats(pipeline_times),
            "inference_latency": latency_stats(inference_times),
            "iterator_wall_latency": latency_stats(wall_times),
            "detections_by_class": dict(sorted(class_counts.items())),
            "detections_by_dataset": {
                dataset: dict(sorted(counts.items()))
                for dataset, counts in sorted(dataset_counts.items())
            },
            "overlay_directory": str(model_output),
        }
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    comparison_csv = output_root / "matched_predictions.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_source_manifest": str(manifest),
        "image_count": len(source_rows),
        "confidence_threshold": args.confidence,
        "imgsz": args.imgsz,
        "batch": 1,
        "device": args.device,
        "models": summaries,
        "matched_predictions_csv": str(comparison_csv),
        "accuracy_limit": (
            "KITTI, CARLA and Oxford images lack labels in a common class schema. "
            "Their outputs measure latency, coverage and qualitative transfer, not accuracy. "
            "COCO and the 21-class Mapillary model have different vocabularies, so raw "
            "detection counts must not be interpreted as a model ranking."
        ),
    }
    summary_path = output_root / "tradeoff_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
