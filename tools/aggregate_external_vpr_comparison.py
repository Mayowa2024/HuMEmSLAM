#!/usr/bin/env python3
"""Aggregate native-model VPR comparison summaries into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "kitti06_clean": "KITTI 06 clean",
    "kitti06_b15_d80": "KITTI 06 B15/D80",
    "kitti06_b35_d80": "KITTI 06 B35/D80",
    "business_fall": "4Seasons Business fall",
    "garage_feb2021": "4Seasons Garage Feb 2021",
    "carla_extreme_rain_fog": "CARLA extreme rain + fog",
    "carla_deep_night": "CARLA deep night",
    "carla_dense_fog_overcast": "CARLA dense fog + overcast",
    "carla_extreme_sunset_glare": "CARLA extreme sunset glare",
    "carla_clear_noon_repeat": "CARLA clear-noon repeat",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("*/*/summary.json")):
        data = json.loads(path.read_text())
        rows.append({
            "dataset": data["dataset"],
            "method": data["method"],
            "queries": data["evaluated_queries"],
            "database_frames": data["database_frames"],
            "recall@1": data["recall@1"],
            "recall@5": data["recall@5"],
            "mrr": data["mrr"],
            "empirical_recall_at_100_percent_precision": data[
                "empirical_recall_at_100_percent_precision"
            ],
            "latency_mean_ms": data["latency_mean_ms"],
            "latency_median_ms": data["latency_median_ms"],
            "latency_p95_ms": data["latency_p95_ms"],
            "descriptor_latency_mean_ms": data.get(
                "descriptor_latency_mean_ms", data["latency_mean_ms"]
            ),
            "descriptor_latency_p95_ms": data.get(
                "descriptor_latency_p95_ms", data["latency_p95_ms"]
            ),
            "end_to_end_latency_mean_ms": data.get(
                "end_to_end_latency_mean_ms", data["latency_mean_ms"]
            ),
            "end_to_end_latency_p95_ms": data.get(
                "end_to_end_latency_p95_ms", data["latency_p95_ms"]
            ),
            "latency_scope": data.get("latency_scope", "not recorded"),
        })
    if not rows:
        raise SystemExit("No summary.json files found")
    with (args.results / "aggregate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    lines = [
        "# External visual-place-recognition comparison",
        "",
        "SALAD and MegaLoc use their official native PyTorch releases. SALAD",
        "does not use the repository's TensorRT or ONNX exports. Every method",
        "uses the same deterministic database and query manifest per dataset.",
        "Accuracy is one deterministic pass; latency is three passes over 30",
        "queries on the RTX 4070 Laptop GPU.",
        "",
        "| Dataset | Method | Database | Queries | R@1 | R@5 | MRR | Empirical recall at 100% precision | Descriptor mean (ms) | Descriptor P95 (ms) | End-to-end mean (ms) | End-to-end P95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {LABELS.get(row['dataset'], row['dataset'])} | "
            f"{row['method'].upper()} | {row['database_frames']} | "
            f"{row['queries']} | {row['recall@1']:.3f} | "
            f"{row['recall@5']:.3f} | {row['mrr']:.3f} | "
            f"{row['empirical_recall_at_100_percent_precision']:.3f} | "
            f"{row['descriptor_latency_mean_ms']:.2f} | "
            f"{row['descriptor_latency_p95_ms']:.2f} | "
            f"{row['end_to_end_latency_mean_ms']:.2f} | "
            f"{row['end_to_end_latency_p95_ms']:.2f} |"
        )
    lines += [
        "",
        "The 100%-precision value is an empirical same-set ceiling, not yet a",
        "held-out operating point. A threshold must be calibrated on a validation",
        "sequence and applied unchanged to the test sequences before making a",
        "general rejection-performance claim.",
        "",
        "The timing boundaries are now standardised. Descriptor latency begins with",
        "an already decoded image and includes model-specific preprocessing, descriptor",
        "extraction and normalisation. End-to-end latency continues through database",
        "similarity and top-5 ranking. HuMemSLAM's end-to-end path additionally contains",
        "the object, OCR and semantic-fusion operations that define the method. Image I/O",
        "and database construction are excluded for every method.",
        "",
    ]
    (args.results / "RESULTS.md").write_text("\n".join(lines))
    print(args.results / "RESULTS.md")


if __name__ == "__main__":
    main()
