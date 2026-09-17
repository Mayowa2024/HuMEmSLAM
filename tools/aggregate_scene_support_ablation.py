#!/usr/bin/env python3
"""Aggregate the 40-run scene-support full-SLAM ablation campaign."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "test_results/dissertation_final/ablations/scene_support_cross_dataset_40x_20260826"
CONFIGS = ("scene_only", "scene_object", "scene_text", "full")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as stream:
        return json.load(stream)


def nested(data: dict, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def latency(path: Path, column: str, skip_first: bool = False):
    if not path.exists():
        return None, None, None, 0
    values = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = number(row.get(column))
            if value is not None:
                values.append(value)
    if skip_first and values:
        values = values[1:]
    if not values:
        return None, None, None, 0
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return statistics.fmean(values), statistics.median(values), p95, len(values)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def extract(dataset: str, config: str) -> dict:
    run = CAMPAIGN / dataset / config / "run_01" / "humanslam"
    retrieval = load_json(run / "loop_retrieval_metrics.json")
    source = nested(retrieval, "sources", "semantic", default={})
    if not source:
        source = retrieval

    run_summary = load_json(run / "run_summary.json")
    trajectory = load_json(run / "trajectory_evaluation/summary.json")
    if not trajectory:
        trajectory = nested(run_summary, "trajectory_evaluation", default={}) or {}
    relocal = load_json(run / "relocalisation_metrics.json")
    if not relocal:
        relocal = nested(run_summary, "relocalisation", default={}) or {}
    aliasing = load_json(run / "aliasing_evaluation/summary.json")

    human_mean, human_median, human_p95, human_n = latency(
        run / "human_latency.csv", "total_ms", skip_first=True
    )
    orb_file = run / "orb_latency.csv"
    if not orb_file.exists():
        orb_file = run / "orb_events_latency.csv"
    orb_mean, _, _, orb_n = latency(orb_file, "track_stereo_ms")

    global_valid = trajectory.get("global_ape_valid")
    maps = trajectory.get("maps", {})
    ape = None
    if global_valid and len(maps) == 1:
        ape = number(nested(next(iter(maps.values())), "ape", "rmse"))

    geometric_attempts = nested(retrieval, "geometric_verification", "attempts")
    accepted = nested(retrieval, "geometric_verification", "accepted_seeds")
    closures = nested(retrieval, "loop_correction", "closures")
    if accepted is None:
        accepted = aliasing.get("accepted")

    tracking = nested(relocal, "tracking_loss", default={}) or {}
    reloc = nested(relocal, "relocalisation", default={}) or {}
    return {
        "dataset": dataset,
        "configuration": config,
        "eligible_queries": source.get("eligible_revisit_queries"),
        "recall_at_1": number(source.get("recall@1")),
        "recall_at_5": number(source.get("recall@5")),
        "mrr": number(source.get("mrr")),
        "geometry_attempts": geometric_attempts,
        "accepted_proposals": accepted,
        "applied_closures": closures,
        "valid_overlap_acceptances": aliasing.get("valid_place_overlap"),
        "cross_floor_proposals": aliasing.get("cross_floor_proposals"),
        "accepted_cross_floor": aliasing.get("accepted_cross_floor"),
        "global_ape_valid": global_valid,
        "ape_rmse_m": ape,
        "tracking_completeness": number(trajectory.get("tracking_completeness")),
        "map_count": trajectory.get("map_count"),
        "loss_episodes": tracking.get("episodes"),
        "recovered_episodes": tracking.get("recovered_episodes"),
        "correct_relocalisations": reloc.get("ground_truth_correct"),
        "human_steady_mean_ms": human_mean,
        "human_steady_median_ms": human_median,
        "human_steady_p95_ms": human_p95,
        "human_latency_samples": human_n,
        "orb_tracking_mean_ms": orb_mean,
        "orb_tracking_samples": orb_n,
        "result_directory": str(run.relative_to(ROOT)),
    }


def macro(rows, config):
    selected = [row for row in rows if row["configuration"] == config]
    def mean(field):
        values = [row[field] for row in selected if number(row[field]) is not None]
        return statistics.fmean(values) if values else None
    return {
        "configuration": config,
        "datasets": len(selected),
        "retrieval_defined_datasets": sum(row["recall_at_1"] is not None for row in selected),
        "macro_recall_at_1": mean("recall_at_1"),
        "macro_recall_at_5": mean("recall_at_5"),
        "macro_mrr": mean("mrr"),
        "mean_human_steady_ms": mean("human_steady_mean_ms"),
        "mean_orb_tracking_ms": mean("orb_tracking_mean_ms"),
        "valid_global_ape_runs": sum(row["global_ape_valid"] is True for row in selected),
        "mean_valid_ape_rmse_m": mean("ape_rmse_m"),
        "total_accepted_proposals": sum(row["accepted_proposals"] or 0 for row in selected),
        "total_applied_closures": sum(row["applied_closures"] or 0 for row in selected),
        "total_correct_relocalisations": sum(row["correct_relocalisations"] or 0 for row in selected),
    }


def main():
    datasets = sorted(path.name for path in CAMPAIGN.iterdir() if path.is_dir())
    rows = [extract(dataset, config) for config in CONFIGS for dataset in datasets]
    output_csv = CAMPAIGN / "aggregate_results.csv"
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    macros = [macro(rows, config) for config in CONFIGS]
    macro_csv = CAMPAIGN / "configuration_summary.csv"
    with macro_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macros[0]))
        writer.writeheader()
        writer.writerows(macros)

    lines = [
        "# Scene-support full-SLAM ablation — aggregate results",
        "",
        "Campaign: `scene_support_cross_dataset_40x_20260826` (40/40 runs complete).",
        "Each cell is one full HuMemSLAM run; results are not repeated-run means.",
        "Human latency is the steady mean after excluding the first recorded query.",
        "A dash means the runner did not emit that metric; it never means zero.",
        "",
        "> **Validity warning:** these configurations were executed as separate online SLAM",
        "> runs. ORB-SLAM3 therefore generated different query frames, keyframes, databases",
        "> and eligible-query denominators. The retrieval columns describe each operational",
        "> run but are not a controlled head-to-head fusion ablation. Do not use the macro",
        "> Recall/MRR ordering to claim that one layer combination ranks better than another.",
        "> Fusion contribution must be measured by rescoring a frozen shared query/database",
        "> manifest with all four configurations.",
        "",
        "## Configuration-level macro summary",
        "",
        "| Configuration | Runs | Retrieval-defined datasets | Macro R@1 | Macro R@5 | Macro MRR | Human steady mean (ms) | ORB tracking mean (ms) | Valid global APE | Mean valid APE (m) | Accepted proposals | Applied closures | Correct relocalisations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macros:
        lines.append(
            f"| {row['configuration']} | {row['datasets']} | {row['retrieval_defined_datasets']}/10 | {fmt(row['macro_recall_at_1'])} | "
            f"{fmt(row['macro_recall_at_5'])} | {fmt(row['macro_mrr'])} | "
            f"{fmt(row['mean_human_steady_ms'])} | {fmt(row['mean_orb_tracking_ms'])} | "
            f"{row['valid_global_ape_runs']}/10 | {fmt(row['mean_valid_ape_rmse_m'])} | "
            f"{row['total_accepted_proposals']} | {row['total_applied_closures']} | "
            f"{row['total_correct_relocalisations']} |"
        )

    lines += ["", "## Per-dataset results", ""]
    for dataset in datasets:
        lines += [
            f"### {dataset}", "",
            "| Configuration | Eligible | R@1 | R@5 | MRR | Accepted / attempts | Closures | APE RMSE (m) | Completeness | Maps | Loss / recovered | Correct reloc. | Human steady mean / p95 (ms) | ORB mean (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [item for item in rows if item["dataset"] == dataset]:
            attempts = row["geometry_attempts"]
            accepted = row["accepted_proposals"]
            accepted_attempts = "—" if accepted is None else str(accepted)
            if attempts is not None:
                accepted_attempts += f" / {attempts}"
            completeness = None if row["tracking_completeness"] is None else 100 * row["tracking_completeness"]
            completeness_text = "—" if completeness is None else f"{completeness:.3f}%"
            lines.append(
                f"| {row['configuration']} | {row['eligible_queries'] if row['eligible_queries'] is not None else '—'} | "
                f"{fmt(row['recall_at_1'])} | {fmt(row['recall_at_5'])} | {fmt(row['mrr'])} | "
                f"{accepted_attempts} | {row['applied_closures'] if row['applied_closures'] is not None else '—'} | "
                f"{fmt(row['ape_rmse_m'])} | {completeness_text} | "
                f"{row['map_count'] if row['map_count'] is not None else '—'} | "
                f"{row['loss_episodes'] if row['loss_episodes'] is not None else '—'} / "
                f"{row['recovered_episodes'] if row['recovered_episodes'] is not None else '—'} | "
                f"{row['correct_relocalisations'] if row['correct_relocalisations'] is not None else '—'} | "
                f"{fmt(row['human_steady_mean_ms'])} / {fmt(row['human_steady_p95_ms'])} | "
                f"{fmt(row['orb_tracking_mean_ms'])} |"
            )
        lines.append("")

    lines += [
        "## Interpretation boundary",
        "",
        "The macro retrieval values are unweighted means over datasets with a defined metric.",
        "The retrieval-defined count must accompany each mean because severe runs with no",
        "eligible query are undefined rather than zero. These summaries do not replace",
        "dataset-level analysis. APE is averaged",
        "only over runs with a valid single-map global trajectory, so configurations with",
        "different valid-run counts are not strictly head-to-head on that column.",
        "Although scene-support is non-negative for any fixed candidate, it can alter ranking",
        "by boosting competing candidates differently. More importantly here, the candidates",
        "and denominators themselves changed between independent SLAM runs.",
        "The 4Seasons campaign runner did not emit the same global trajectory summary as the",
        "KITTI/CARLA runner; its unavailable trajectory fields remain blank.",
        "",
        "Machine-readable files: `aggregate_results.csv` and `configuration_summary.csv`.",
    ]
    (CAMPAIGN / "AGGREGATE_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
