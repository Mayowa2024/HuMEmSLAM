#!/usr/bin/env python3
"""Export the frozen HuMemSLAM YOLO segmentation checkpoint to TensorRT."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "weights/humanSLAM_YOLO_seg.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot access a CUDA device")

    model = YOLO(str(weights), task="segment")
    exported = Path(
        model.export(
            format="engine",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            dynamic=False,
            workspace=args.workspace,
            simplify=True,
        )
    ).resolve()
    expected = weights.with_suffix(".engine")
    if exported != expected:
        raise RuntimeError(f"Unexpected export path: {exported}; expected {expected}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(weights),
        "source_sha256": sha256(weights),
        "engine": str(exported),
        "engine_sha256": sha256(exported),
        "engine_size_bytes": exported.stat().st_size,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "dynamic": False,
        "workspace_gib": args.workspace,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    metadata_path = exported.with_suffix(".engine.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
