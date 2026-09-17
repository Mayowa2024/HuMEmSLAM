#!/usr/bin/env python3
"""Build the fixed per-dataset ViT-B VLAD vocabulary from reference patches."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
RA = ROOT / "external/vpr_comparison/Revisit-Anything"
sys.path.insert(0, str(RA))
from place_rec_global_config import datasets, workdir_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-patches", type=int, default=100_000)
    args = parser.parse_args()

    config = datasets[args.dataset]
    cfg = config["cfg"]
    h5_path = Path(workdir_data) / args.dataset / "out" / config["dino_h5_filename_r"]
    rng = np.random.default_rng(20260821)
    batches = []
    seen = 0
    with h5py.File(h5_path, "r") as h5:
        for key in sorted(h5.keys()):
            feat = h5[key]["ift_dino"][()][0].reshape(cfg["desc_dim"], -1).T
            remaining = args.max_patches - seen
            if remaining <= 0:
                break
            if len(feat) > remaining:
                feat = feat[rng.choice(len(feat), remaining, replace=False)]
            batches.append(feat.astype(np.float32, copy=False))
            seen += len(feat)
    samples = np.concatenate(batches, axis=0)
    samples /= np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1e-12)
    kmeans = MiniBatchKMeans(
        n_clusters=32, random_state=20260821, batch_size=4096,
        n_init=3, max_iter=200,
    ).fit(samples)
    output = (
        RA / "cache/vocabulary" / cfg["dino_model"] /
        f"l{cfg['dino_layer']}_value_c32" / config["domain_vlad_cluster"] /
        "c_centers.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)), output)
    print(f"Saved {output} from {len(samples)} reference patches")


if __name__ == "__main__":
    main()
