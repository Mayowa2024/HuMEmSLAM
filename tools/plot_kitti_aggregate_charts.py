#!/usr/bin/env python3
"""Create simple, consistent aggregate dissertation charts for KITTI 00/05/06."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "test_results/dissertation_final/kitti"
OUTPUT = RESULTS / "aggregate_charts"
COLOURS = {"baseline": "#4C78A8", "humanslam": "#F58518"}
LABELS = {"baseline": "ORB-SLAM3", "humanslam": "HuMemSLAM"}


def load(path):
    return json.loads(path.read_text())


def nested(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict): return default
        value = value.get(key)
    return default if value is None else value


def aggregate(sequence, specs):
    output = []
    for condition, label, directory in specs:
        for mode, source in (("baseline", "native_bow"), ("humanslam", "semantic")):
            rows = []
            for run in sorted(directory.glob("run_*")):
                p = run / mode
                try:
                    summary = load(p / "run_summary.json")
                    retrieval = load(p / "loop_retrieval_metrics.json")
                    relocal = load(p / "relocalisation_metrics.json")
                except FileNotFoundError:
                    continue
                trajectory = summary.get("trajectory_evaluation", {})
                maps = trajectory.get("maps", {})
                valid = (bool(trajectory.get("global_ape_valid")) and len(maps) == 1
                         and float(trajectory.get("tracking_completeness") or 0) >= .99)
                ape = next(iter(maps.values()), {}).get("ape", {}).get("rmse") if valid else None
                src = nested(retrieval, "sources", source, default={})
                rows.append({
                    "r1": src.get("recall@1"), "r3": src.get("recall@3"),
                    "r5": src.get("recall@5"), "ape": ape,
                    "complete": trajectory.get("tracking_completeness"),
                    "closures": nested(retrieval, "loop_correction", "closures", default=0),
                    "attempts": nested(retrieval, "geometric_verification", "attempts", default=0),
                    "correct_relocal": nested(relocal, "relocalisation", "ground_truth_correct", default=0),
                    "orb_ms": nested(summary, "orb_tracking_ms", "mean"),
                    "human_ms": nested(summary, "human_total_ms", "mean"),
                })
            def vals(key): return [float(row[key]) for row in rows if row.get(key) is not None]
            def mean(key):
                values = vals(key); return float(np.mean(values)) if values else None
            apes = vals("ape")
            output.append({
                "sequence": sequence, "condition": condition, "label": label,
                "mode": mode, "runs": len(rows),
                "recall1": mean("r1"), "recall3": mean("r3"), "recall5": mean("r5"),
                "ape_mean": float(np.mean(apes)) if apes else None,
                "ape_sd": float(np.std(apes, ddof=1)) if len(apes) > 1 else 0.0 if apes else None,
                "valid_ape_runs": len(apes), "completeness": mean("complete"),
                "closures": int(sum(row["closures"] for row in rows)),
                "correct_relocal": int(sum(row["correct_relocal"] for row in rows)),
                "geometry_attempts": int(sum(row["attempts"] for row in rows)),
                "orb_ms": mean("orb_ms"), "human_ms": mean("human_ms"),
            })
    return output


def style(ax, labels):
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)


def save(fig, directory, name):
    fig.tight_layout(); fig.savefig(directory / f"{name}.png", dpi=200)
    fig.savefig(directory / f"{name}.pdf"); plt.close(fig)


def charts(sequence, rows):
    directory = OUTPUT / sequence; directory.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    labels = [next(row["label"] for row in rows if row["condition"] == item) for item in conditions]
    by = {(row["condition"], row["mode"]): row for row in rows}
    x = np.arange(len(conditions)); width = .36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, ("baseline", "humanslam")):
        for metric, marker in (("recall1", "o"), ("recall3", "s"), ("recall5", "^")):
            ax.plot(x, [by[c, mode][metric] if by[c, mode][metric] is not None else np.nan
                        for c in conditions], marker=marker, linewidth=2,
                    label=metric.replace("recall", "Recall@"))
        ax.set_title(LABELS[mode]); ax.set_ylim(0, 1.02); ax.set_ylabel("Recall")
        style(ax, labels); ax.legend(frameon=False)
    fig.suptitle(f"KITTI {sequence}: place-retrieval recall")
    save(fig, directory, "01_retrieval_recall")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, mode in ((-width/2, "baseline"), (width/2, "humanslam")):
        values=[by[c,mode]["ape_mean"] if by[c,mode]["ape_mean"] is not None else 0 for c in conditions]
        errors=[by[c,mode]["ape_sd"] or 0 for c in conditions]
        bars=ax.bar(x+offset,values,width,yerr=errors,capsize=3,label=LABELS[mode],color=COLOURS[mode])
        for bar,c in zip(bars,conditions):
            n=by[c,mode]["valid_ape_runs"]
            if n == 0: ax.text(bar.get_x()+bar.get_width()/2,.03,"invalid",ha="center",va="bottom",rotation=90,fontsize=8)
            else: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+errors[conditions.index(c)]+.03,f"n={n}",ha="center",fontsize=8)
    ax.set_ylabel("Global APE RMSE (m), mean ± SD"); ax.legend(frameon=False); style(ax,labels)
    ax.set_title(f"KITTI {sequence}: trajectory accuracy (valid single-map runs only)")
    save(fig,directory,"02_ape_rmse")

    fig, ax = plt.subplots(figsize=(9,4.8))
    for offset,mode in ((-width/2,"baseline"),(width/2,"humanslam")):
        ax.bar(x+offset,[by[c,mode]["closures"] for c in conditions],width,
               label=LABELS[mode],color=COLOURS[mode])
    ax.set_ylabel("Applied loop closures (total across runs)")
    ax.set_title(f"KITTI {sequence}: applied loop closures")
    style(ax,labels); ax.legend(frameon=False)
    save(fig,directory,"03_applied_loop_closures")

    failure_conditions = [c for c in conditions if "clean" not in
                          next(row["label"] for row in rows
                               if row["condition"] == c).lower() and (
                          by[c,"baseline"]["correct_relocal"] > 0 or
                          by[c,"humanslam"]["correct_relocal"] > 0)]
    if failure_conditions:
        failure_labels = [next(row["label"] for row in rows
                               if row["condition"] == c) for c in failure_conditions]
        fx = np.arange(len(failure_conditions))
        fig, ax = plt.subplots(figsize=(max(6, 2.2*len(failure_conditions)),4.8))
        for offset,mode in ((-width/2,"baseline"),(width/2,"humanslam")):
            bars=ax.bar(fx+offset,[by[c,mode]["correct_relocal"] for c in failure_conditions],
                        width,label=LABELS[mode],color=COLOURS[mode])
            ax.bar_label(bars,fmt="%d",padding=3)
        ax.set_ylabel("Ground-truth-correct relocalisations (total)")
        ax.set_title(f"KITTI {sequence}: relocalisation in conditions with observed recovery")
        ax.set_xticks(fx,failure_labels,rotation=20,ha="right")
        ax.grid(axis="y",alpha=.25); ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.text(.5,.01,"Only perturbed conditions with at least one successful relocalisation are shown.",ha="center",fontsize=9)
        save(fig,directory,"03b_failure_only_relocalisation")

    fig,ax=plt.subplots(figsize=(9,4.8))
    for offset,mode in ((-width/2,"baseline"),(width/2,"humanslam")):
        ax.bar(x+offset,[by[c,mode]["geometry_attempts"] for c in conditions],width,label=LABELS[mode],color=COLOURS[mode])
    ax.set_ylabel("Geometry-verification attempts (total)"); ax.set_title(f"KITTI {sequence}: verification workload")
    ax.legend(frameon=False); style(ax,labels); save(fig,directory,"04_geometry_workload")

    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for offset,mode in ((-width/2,"baseline"),(width/2,"humanslam")):
        axes[0].bar(x+offset,[100*by[c,mode]["completeness"] for c in conditions],width,label=LABELS[mode],color=COLOURS[mode])
        axes[1].bar(x+offset,[by[c,mode]["orb_ms"] for c in conditions],width,label=LABELS[mode],color=COLOURS[mode])
    axes[0].set_ylabel("Mean tracking completeness (%)"); axes[0].set_ylim(0,101)
    axes[1].set_ylabel("Mean ORB TrackStereo latency (ms)")
    for ax in axes: style(ax,labels); ax.legend(frameon=False)
    fig.suptitle(f"KITTI {sequence}: continuity and tracking latency")
    save(fig,directory,"05_completeness_and_latency")

    with (directory/"aggregate_chart_data.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    specs = {
        "06": [(p.name.split("_",1)[0], p.name.split("_",1)[1].replace("_"," "), p/"formal_10x")
               for p in sorted(RESULTS.glob("K06-*")) if (p/"formal_10x").is_dir()],
        "00": [("K00-0","Clean",RESULTS/"K00-0_clean"),("K00-1","B15/D80",RESULTS/"K00-1_blur15_dark80")],
        "05": [("K05-0","Clean",RESULTS/"K05-0_clean"),("K05-1","B15/D80",RESULTS/"K05-1_blur15_dark80")],
    }
    for sequence, items in specs.items(): charts(sequence, aggregate(sequence,items))
    print(OUTPUT)


if __name__ == "__main__": main()
