#!/usr/bin/env python3
"""Reproducible YOLO26 segmentation fine-tuning for HuMemSLAM."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "training_data/mapillary_humanslam_full/mapillary_humanslam.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "training_data/pretrained/yolo26n-seg.pt",
    )
    parser.add_argument("--project", type=Path, default=root / "runs/mapillary_humanslam")
    parser.add_argument("--name", default="yolo26n_seg_full_seed42_b8")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {args.data}")
    if not args.weights.is_file() and args.resume is None:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.weights}")
    if str(args.device) != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it")
        torch.zeros(1, device="cuda")

    run_dir = args.project.resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command_parameters": vars(args),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    metadata["command_parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["command_parameters"].items()
    }
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if args.resume:
        model = YOLO(str(args.resume.resolve()))
        model.train(resume=True)
        return

    model = YOLO(str(args.weights.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=False,
        save_period=args.save_period,
        plots=True,
    )


if __name__ == "__main__":
    main()
