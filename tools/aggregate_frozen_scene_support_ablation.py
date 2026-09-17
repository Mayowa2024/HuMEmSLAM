#!/usr/bin/env python3
"""Aggregate the controlled frozen-evidence scene-support ablation."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "test_results/dissertation_final/ablations/frozen_scene_support_10datasets_20260826"
CONFIGS = ("scene_only", "scene_object", "scene_text", "full")


def f(value):
    return f"{float(value):.3f}"


def main():
    rows = []
    for path in sorted(CAMPAIGN.glob("*/summary.csv")):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))

    with (CAMPAIGN / "aggregate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summaries = []
    for config in CONFIGS:
        selected = [row for row in rows if row["configuration"] == config]
        summaries.append({
            "configuration": config,
            "datasets": len(selected),
            "queries": sum(int(row["queries"]) for row in selected),
            "macro_recall_at_1": statistics.fmean(float(row["recall@1"]) for row in selected),
            "macro_recall_at_5": statistics.fmean(float(row["recall@5"]) for row in selected),
            "macro_mrr": statistics.fmean(float(row["mrr"]) for row in selected),
            "rank1_changes": sum(int(row["rank1_changes_vs_scene_only"]) for row in selected),
            "beneficial_changes": sum(int(row["beneficial_rank1_changes"]) for row in selected),
            "harmful_changes": sum(int(row["harmful_rank1_changes"]) for row in selected),
            "perception_mean_ms": statistics.fmean(float(row["perception_mean_ms"]) for row in selected),
            "ranking_mean_ms": statistics.fmean(float(row["ranking_mean_ms"]) for row in selected),
            "end_to_end_mean_ms": statistics.fmean(float(row["end_to_end_mean_ms"]) for row in selected),
        })
    with (CAMPAIGN / "configuration_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)

    lines = [
        "# Controlled frozen-evidence scene-support ablation",
        "",
        "All four configurations rescore identical cached semantic records, database frames,",
        "query frames and ground-truth positives. Each of the ten subsets contains 100 queries.",
        "This is the valid retrieval ablation; the independent full-SLAM runs remain an",
        "operational-outcome study rather than a head-to-head ranking comparison.",
        "",
        "## Aggregate across 10 datasets / 1,000 shared queries",
        "",
        "| Configuration | R@1 | R@5 | MRR | Top-1 changes | Beneficial | Harmful | Perception (ms) | Ranking (ms) | End-to-end (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['configuration']} | {f(row['macro_recall_at_1'])} | {f(row['macro_recall_at_5'])} | "
            f"{f(row['macro_mrr'])} | {row['rank1_changes']} | {row['beneficial_changes']} | "
            f"{row['harmful_changes']} | {f(row['perception_mean_ms'])} | "
            f"{f(row['ranking_mean_ms'])} | {f(row['end_to_end_mean_ms'])} |"
        )
    lines += ["", "## Per-dataset results", "",
              "| Dataset | Configuration | R@1 | R@5 | MRR | Top-1 changes | Beneficial | Harmful | End-to-end mean / p95 (ms) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['configuration']} | {f(row['recall@1'])} | "
            f"{f(row['recall@5'])} | {f(row['mrr'])} | {row['rank1_changes_vs_scene_only']} | "
            f"{row['beneficial_rank1_changes']} | {row['harmful_rank1_changes']} | "
            f"{f(row['end_to_end_mean_ms'])} / {f(row['end_to_end_p95_ms'])} |"
        )
    lines += [
        "", "## Interpretation", "",
        "A support layer is beneficial when it changes an incorrect scene-only top-1 into a",
        "correct top-1, and harmful when it changes a correct scene-only top-1 into an incorrect",
        "one. Neutral rank changes preserve top-1 correctness. Non-negative support can still",
        "harm ranking because different candidates receive different support boosts.", "",
        "Latency excludes image decoding and database construction. Perception is executed for",
        "the enabled layers, while ranking uses the frozen cached records.",
    ]
    (CAMPAIGN / "AGGREGATE_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
