#!/usr/bin/env python3
"""Standardised three-pass query timing for official Revisit Anything."""

import argparse, csv, gc, json, os, pickle, sys, time
from pathlib import Path

import cv2, faiss, h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
RA = ROOT / "external/vpr_comparison/Revisit-Anything"
sys.path.insert(0, str(RA)); os.chdir(RA)
import func_vpr  # noqa: E402
from place_rec_global_config import datasets  # noqa: E402
from run_external_vpr_comparison import dataset_definition, evenly_select  # noqa: E402


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True)
    p.add_argument("--output-root", required=True, type=Path); p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args(); data = dataset_definition(args.dataset, 100)
    stage = RA / "workdir_data" / args.dataset / "out"
    output = args.output_root / args.dataset / "revisit_anything"; output.mkdir(parents=True, exist_ok=True)
    timed = evenly_select(data["query_frames"], 30)
    query_index = {frame: index for index, frame in enumerate(data["query_frames"])}
    images = sorted((RA / "workdir_data" / args.dataset / "query").iterdir())

    cfg = {**datasets[args.dataset]["cfg"], "resize": True, "dinov2": True}
    sam_cfg = {"desired_width": cfg["desired_width"], "desired_height": cfg["desired_height"],
        "resize": True, "min_mask_area_ratio": cfg["min_mask_area_ratio"], "max_masks": cfg["max_masks"]}
    sam = func_vpr.loadSAM_EfficientViT(
        str(ROOT / "external/vpr_comparison/efficientvit/assets/checkpoints/efficientvit_sam/efficientvit_sam_l2.pt"),
        sam_cfg, "cuda")
    sam_times = {}
    for repetition in range(1, args.repetitions + 1):
        for frame in timed:
            image = cv2.imread(str(data["query_paths"][frame]))
            started = time.perf_counter(); func_vpr.process_single_SAM(sam_cfg, image, sam, "cuda")
            sam_times[(repetition, frame)] = (time.perf_counter() - started) * 1000
    del sam; gc.collect(); torch.cuda.empty_cache()

    dino = func_vpr.loadDINO(cfg, "cuda")
    centers = torch.load(RA / "cache/vocabulary" / cfg["dino_model"] /
        f"l{cfg['dino_layer']}_value_c32" / args.dataset / "c_centers.pt")
    with open(stage / f"{args.dataset}_r_fitted_pca_model_order3.pkl", "rb") as stream: pca = pickle.load(stream)
    with open(stage / "fixed_manifest_bundle.pkl", "rb") as stream: bundle = pickle.load(stream)
    reference = norm(bundle["reference_descriptors"].numpy()); owners = np.asarray(bundle["reference_image_indices"], int)
    height, width = cfg["desired_height"], cfg["desired_width"]
    dh, dw = height // 14, width // 14
    idx = np.empty((height, width, 2), np.int32)
    for i in range(height):
        for j in range(width): idx[i, j] = [min(i // 14, dh - 1), min(j // 14, dw - 1)]
    ind = torch.tensor(np.ravel_multi_index(idx.reshape(-1, 2).T, (dh, dw)), device="cuda")
    masks_file = stage / f"{args.dataset}_q_masks_320.h5"
    latency = []
    with h5py.File(masks_file, "r") as masks_h5:
        for repetition in range(1, args.repetitions + 1):
            for frame in timed:
                qi = query_index[frame]; key = images[qi].name
                masks = func_vpr.preload_masks(masks_h5, key)
                image = cv2.imread(str(data["query_paths"][frame]))
                started = time.perf_counter()
                _, feature = func_vpr.process_single_DINO(cfg, image, dino, "cuda")
                adjacency = func_vpr.nbrMasksAGGFastSingle(masks, 3)
                segment = func_vpr.seg_vlad_gpu_single_img(ind, idx, feature.cpu(), key, masks, centers, cfg, cfg["desc_dim"], adjacency)
                descriptor = norm(pca.transform(segment.numpy()))
                remainder_ms = (time.perf_counter() - started) * 1000
                search_started = time.perf_counter()
                allowed_images = [i for i, f in enumerate(data["database_frames"]) if f <= frame - 100]
                allowed_segments = np.flatnonzero(np.isin(owners, allowed_images))
                index = faiss.IndexFlatL2(reference.shape[1]); index.add(reference[allowed_segments])
                index.search(descriptor, min(50, len(allowed_segments)))
                search_ms = (time.perf_counter() - search_started) * 1000
                descriptor_ms = sam_times[(repetition, frame)] + remainder_ms
                latency.append({"repetition": repetition, "query_frame": frame,
                    "descriptor_ms": descriptor_ms, "search_ms": search_ms,
                    "end_to_end_ms": descriptor_ms + search_ms})
    with (output / "latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latency[0])); writer.writeheader(); writer.writerows(latency)
    desc = np.asarray([r["descriptor_ms"] for r in latency]); end = np.asarray([r["end_to_end_ms"] for r in latency])
    summary = json.loads((output / "accuracy_summary.json").read_text())
    summary.update({"latency_repetitions": args.repetitions, "latency_queries_per_repetition": len(timed),
        "descriptor_latency_mean_ms": float(desc.mean()), "descriptor_latency_median_ms": float(np.median(desc)),
        "descriptor_latency_p95_ms": float(np.percentile(desc, 95)),
        "end_to_end_latency_mean_ms": float(end.mean()), "end_to_end_latency_median_ms": float(np.median(end)),
        "end_to_end_latency_p95_ms": float(np.percentile(end, 95)), "latency_mean_ms": float(end.mean()),
        "latency_median_ms": float(np.median(end)), "latency_p95_ms": float(np.percentile(end, 95)),
        "latency_scope": "decoded image -> EfficientViT-SAM-L2 -> DINOv2 ViT-B/14 -> SegVLAD/PCA -> constrained segment search -> top-5"})
    summary.pop("latency_status", None); (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
