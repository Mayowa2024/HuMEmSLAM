#!/usr/bin/env python3
"""Ground-truth validation of the Mapillary PT and TensorRT runtimes."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root
        / "training_data/mapillary_humanslam_full/mapillary_humanslam.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "validation_results/yolo_model_tradeoff/mapillary_accuracy.json",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def serialise_metrics(metrics):
    return {
        "results_dict": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "box_maps_by_class": [float(value) for value in metrics.box.maps],
        "mask_maps_by_class": [float(value) for value in metrics.seg.maps],
        "class_names": {str(key): value for key, value in metrics.names.items()},
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.data.resolve()
    specifications = {
        "MAPILLARY_FINETUNED_HUMANSLAM_PT": root / "weights/humanSLAM_YOLO_seg.pt",
        "MAPILLARY_FINETUNED_HUMANSLAM_TENSORRT_FP16": root
        / "weights/humanSLAM_YOLO_seg.engine",
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data),
        "imgsz": args.imgsz,
        "batch": 1,
        "device": args.device,
        "models": {},
        "coco_exclusion": (
            "The COCO checkpoint is not scored against these labels because its 80-class "
            "IDs and meanings do not match the 21-class Mapillary HuMemSLAM schema."
        ),
    }
    for name, model_path in specifications.items():
        model = YOLO(str(model_path.resolve()), task="segment")
        metrics = model.val(
            data=str(data),
            split="val",
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            workers=4,
            plots=False,
            save_json=False,
            verbose=False,
        )
        report["models"][name] = {
            "model_path": str(model_path.resolve()),
            **serialise_metrics(metrics),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
