#!/usr/bin/env python3
"""Run one official Revisit Anything preprocessing stage for one split."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIT = ROOT / "external/vpr_comparison/Revisit-Anything"
sys.path.insert(0, str(REVISIT))
os.chdir(REVISIT)

import func_vpr  # noqa: E402
from place_rec_global_config import datasets, workdir_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("ref", "query"), required=True)
    parser.add_argument("--method", choices=("SAM", "DINO"), required=True)
    args = parser.parse_args()
    config = datasets[args.dataset]
    full = config["cfg"]
    directory = Path(workdir_data) / args.dataset / args.split
    images = sorted(path.name for path in directory.iterdir() if path.is_file() or path.is_symlink())
    out = Path(workdir_data) / args.dataset / "out"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "r" if args.split == "ref" else "q"
    if args.method == "SAM":
        cfg = {
            "desired_width": full["desired_width"],
            "desired_height": full["desired_height"],
            "rmin": 0, "resize": True, "dinov2": True,
            "min_mask_area_ratio": full.get("min_mask_area_ratio", 0.0025),
            "max_masks": full.get("max_masks", 48),
        }
        model = func_vpr.loadSAM_EfficientViT(
            str(ROOT / "external/vpr_comparison/efficientvit/assets/checkpoints/efficientvit_sam/efficientvit_sam_l2.pt"),
            cfg, device="cuda"
        )
        target = out / config[f"masks_h5_filename_{suffix}"]
        func_vpr.process_SAM_to_h5(str(target), cfg, images, model, dataDir=str(directory))
    else:
        cfg = {
            "desired_width": full["desired_width"],
            "desired_height": full["desired_height"],
            "rmin": 0, "resize": True, "dinov2": True,
            "detect": "dino", "use_sam": True, "class_threshold": 0.9,
            "desired_feature": 0, "query_type": "text", "sort_by": "area",
            "use_16bit": False, "use_cuda": True, "dino_strides": 4,
            "use_traced_model": False, "DAStoreFull": False, "wrap": False,
            "dino_model": full.get("dino_model", "dinov2_vitb14"),
            "dino_layer": full.get("dino_layer", 11),
        }
        model = func_vpr.loadDINO(cfg, device="cuda")
        target = out / config[f"dino_h5_filename_{suffix}"]
        func_vpr.process_dino_ft_to_h5(str(target), cfg, images, model, dataDir=str(directory))
    print(target)


if __name__ == "__main__":
    main()
