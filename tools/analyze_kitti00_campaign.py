#!/usr/bin/env python3
"""Aggregate completed KITTI 00 baseline/HuMemSLAM matched campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def metric(rows, key, operation="mean"):
    values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None])
    if not len(values):
        return None
    if operation == "sd":
        return float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean())


def number(value, digits=3):
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: list[dict], aggregates: list[dict]):
    aggregate_by_key = {(row["condition"], row["mode"]): row for row in aggregates}
    lines = [
        "# KITTI 00 complete results sheet", "",
        "Generated from the frozen per-run JSON/CSV artefacts by",
        "`tools/analyze_kitti00_campaign.py`. B15/D80 perturbations apply only",
        "to the four return windows documented in `KITTI_RESULTS_RECORD.md`.", "",
        "## Aggregate results", "",
        "| Condition | Method | n | R@1 | R@5 | MRR | Geometry accepted/attempted | Closures | Loss/LOST/recovered | Correct relocalisations | Maps created/merged | Completeness | APE RMSE mean ± SD (m) | ORB mean (ms) | Human mean/median/p95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("K00-0_clean", "K00-1_blur15_dark80"):
        for mode, label in (("baseline", "ORB-SLAM3"), ("humanslam", "HuMemSLAM")):
            row = aggregate_by_key[(condition, mode)]
            human_latency = "—" if mode == "baseline" else "/".join(
                number(row[key]) for key in
                ("human_total_mean_ms", "human_total_median_ms", "human_total_p95_ms"))
            lines.append(
                f"| {condition} | {label} | {row['runs']} | {number(row['recall1_mean'])} | "
                f"{number(row['recall5_mean'])} | {number(row['mrr_mean'])} | "
                f"{row['geometry_accepted']}/{row['geometry_attempts']} | {row['closures']} | "
                f"{row['loss_episodes']}/{row['lost_episodes']}/{row['recovered_episodes']} | "
                f"{row['gt_correct_relocal']} | {row['maps_created']}/{row['maps_merged']} | "
                f"{number(100 * row['mean_completeness'])}% | "
                f"{number(row['ape_rmse_mean_m'])} ± {number(row['ape_rmse_sd_m'])} | "
                f"{number(row['orb_tracking_mean_ms'])} | {human_latency} |"
            )

    lines += [
        "", "## Per-run results", "",
        "`A/G` is accepted geometric seeds / attempts. `L/F/R` is tracking-loss",
        "episodes / episodes reaching fully `LOST` / recovered episodes. `C/M`",
        "is maps created / merged. A dash for APE means global APE was invalid,",
        "not zero.", "",
    ]
    for condition in ("K00-0_clean", "K00-1_blur15_dark80"):
        lines += [
            f"### {condition}", "",
            "| Run | Method | R@1 | R@3 | R@5 | MRR | A/G | Closures | L/F/R | Correct relocalisations | C/M | Maps remaining | Completeness | APE RMSE (m) | ORB mean (ms) | Human mean/median/p95 (ms) | Status/notes |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        subset = [row for row in rows if row["condition"] == condition]
        for row in subset:
            label = "ORB-SLAM3" if row["mode"] == "baseline" else "HuMemSLAM"
            human_latency = "—" if row["mode"] == "baseline" else "/".join(
                number(row[key]) for key in
                ("human_mean_ms", "human_median_ms", "human_p95_ms"))
            notes = "valid"
            if not row["global_ape_valid"]:
                notes = "global APE invalid: fragmented trajectory"
            elif float(row["completeness"] or 0) < .99:
                notes = "valid with <99% completeness"
            lines.append(
                f"| {row['run'].removeprefix('run_')} | {label} | {number(row['recall1'])} | "
                f"{number(row['recall3'])} | {number(row['recall5'])} | {number(row['mrr'])} | "
                f"{row['geometry_accepted']}/{row['geometry_attempts']} | {row['closures']} | "
                f"{row['loss_episodes']}/{row['lost_episodes']}/{row['recovered_episodes']} | "
                f"{row['gt_correct_relocal']} | {row['maps_created']}/{row['maps_merged']} | "
                f"{row['map_count']} | {number(100 * float(row['completeness']))}% | "
                f"{number(row['ape_rmse_m'])} | {number(row['orb_mean_ms'])} | "
                f"{human_latency} | {notes} |"
            )
        lines.append("")

    lines += [
        "## Interpretation constraints", "",
        "- Clean K00 has no tracking-loss episode; it evaluates proactive retrieval and loop closure, not relocalisation.",
        "- A returned candidate is not a closure until ORB-SLAM3 geometry accepts it and applies a correction.",
        "- All recorded relocalisation successes in these campaigns were ground-truth correct; source attribution is summarised in `KITTI_RESULTS_RECORD.md`.",
        "- K00-1 HuMemSLAM run 09 ended with three maps; its global APE is therefore intentionally omitted.",
        "- Aggregate APE must not be used as the sole measure of retrieval or relocalisation quality.",
        "", "## Machine-readable sources", "",
        "```text",
        "test_results/dissertation_final/kitti/analysis_k00/per_run_metrics.csv",
        "test_results/dissertation_final/kitti/analysis_k00/aggregate_metrics.csv",
        "test_results/dissertation_final/kitti/analysis_k00/aggregate_metrics.json",
        "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "analysis_k00"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in ("K00-0_clean", "K00-1_blur15_dark80"):
        condition_dir = args.root / condition
        for run in sorted(condition_dir.glob("run_*")):
            for mode, source in (("baseline", "native_bow"), ("humanslam", "semantic")):
                directory = run / mode
                required = [directory / name for name in (
                    "run_summary.json", "loop_retrieval_metrics.json",
                    "relocalisation_metrics.json")]
                if not all(path.is_file() for path in required):
                    continue
                summary, retrieval, relocal = map(load, required)
                trajectory = get(summary, "trajectory_evaluation", default={})
                maps = trajectory.get("maps", {})
                ape = next(iter(maps.values()), {}).get("ape", {}) if len(maps) == 1 else {}
                retrieval_source = get(retrieval, "sources", source, default={})
                rows.append({
                    "condition": condition, "run": run.name, "mode": mode,
                    "completeness": trajectory.get("tracking_completeness"),
                    "global_ape_valid": trajectory.get("global_ape_valid"),
                    "map_count": trajectory.get("map_count"),
                    "ape_rmse_m": ape.get("rmse"),
                    "recall1": retrieval_source.get("recall@1"),
                    "recall3": retrieval_source.get("recall@3"),
                    "recall5": retrieval_source.get("recall@5"),
                    "mrr": retrieval_source.get("mrr"),
                    "eligible_queries": retrieval_source.get("eligible_revisit_queries"),
                    "geometry_attempts": get(retrieval, "geometric_verification", "attempts", default=0),
                    "geometry_accepted": get(retrieval, "geometric_verification", "accepted_seeds", default=0),
                    "closures": get(retrieval, "loop_correction", "closures", default=0),
                    "loss_episodes": get(relocal, "tracking_loss", "episodes", default=0),
                    "lost_episodes": get(relocal, "tracking_loss", "episodes_reaching_lost", default=0),
                    "recovered_episodes": get(relocal, "tracking_loss", "recovered_episodes", default=0),
                    "relocal_successes": get(relocal, "relocalisation", "successes", default=0),
                    "gt_correct_relocal": get(relocal, "relocalisation", "ground_truth_correct", default=0),
                    "maps_created": get(relocal, "maps", "created", default=0),
                    "maps_merged": get(relocal, "maps", "merges", default=0),
                    "orb_mean_ms": get(summary, "orb_tracking_ms", "mean"),
                    "human_mean_ms": get(summary, "human_total_ms", "mean"),
                    "human_median_ms": get(summary, "human_total_ms", "median"),
                    "human_p95_ms": get(summary, "human_total_ms", "p95"),
                })

    with (output / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    aggregates = []
    for condition in ("K00-0_clean", "K00-1_blur15_dark80"):
        for mode in ("baseline", "humanslam"):
            subset = [row for row in rows if row["condition"] == condition and row["mode"] == mode]
            aggregates.append({
                "condition": condition, "mode": mode, "runs": len(subset),
                "global_ape_valid_runs": sum(bool(row["global_ape_valid"]) for row in subset),
                "complete_runs_ge_99pct": sum((row["completeness"] or 0) >= .99 for row in subset),
                "mean_completeness": metric(subset, "completeness"),
                "ape_rmse_mean_m": metric(subset, "ape_rmse_m"),
                "ape_rmse_sd_m": metric(subset, "ape_rmse_m", "sd"),
                "recall1_mean": metric(subset, "recall1"),
                "recall3_mean": metric(subset, "recall3"),
                "recall5_mean": metric(subset, "recall5"),
                "mrr_mean": metric(subset, "mrr"),
                "geometry_attempts": int(sum(row["geometry_attempts"] or 0 for row in subset)),
                "geometry_accepted": int(sum(row["geometry_accepted"] or 0 for row in subset)),
                "runs_with_closure": sum((row["closures"] or 0) > 0 for row in subset),
                "closures": int(sum(row["closures"] or 0 for row in subset)),
                "loss_episodes": int(sum(row["loss_episodes"] or 0 for row in subset)),
                "lost_episodes": int(sum(row["lost_episodes"] or 0 for row in subset)),
                "recovered_episodes": int(sum(row["recovered_episodes"] or 0 for row in subset)),
                "relocal_successes": int(sum(row["relocal_successes"] or 0 for row in subset)),
                "gt_correct_relocal": int(sum(row["gt_correct_relocal"] or 0 for row in subset)),
                "maps_created": int(sum(row["maps_created"] or 0 for row in subset)),
                "maps_merged": int(sum(row["maps_merged"] or 0 for row in subset)),
                "orb_tracking_mean_ms": metric(subset, "orb_mean_ms"),
                "human_total_mean_ms": metric(subset, "human_mean_ms"),
                "human_total_median_ms": metric(subset, "human_median_ms"),
                "human_total_p95_ms": metric(subset, "human_p95_ms"),
            })
    with (output / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader(); writer.writerows(aggregates)
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    write_markdown(args.root.parents[2] / "docs" / "KITTI00_RESULTS_SHEET.md", rows, aggregates)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
