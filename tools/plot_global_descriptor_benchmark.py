#!/usr/bin/env python3
"""Create dissertation-ready plots from descriptor condition_summary.csv."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/humanslam_matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "places365_resnet50_embedding": "Places365",
    "eigenplaces_r18_512": "EigenPlaces",
    "mixvpr_r50_512": "MixVPR",
    "salad": "SALAD",
}
COLORS = {
    "places365_resnet50_embedding": "#7f7f7f",
    "eigenplaces_r18_512": "#1f77b4",
    "mixvpr_r50_512": "#2ca02c",
    "salad": "#d62728",
}
CONDITION_ORDER = [
    "original", "dark_50", "dark_80", "blur_15", "blur_35",
    "combo_15_50", "combo_35_80", "low_contrast", "fog",
    "occlusion_25",
]
CONDITION_NAMES = {
    "original": "Original",
    "dark_50": "Dark 50%",
    "dark_80": "Dark 80%",
    "blur_15": "Blur 15 px",
    "blur_35": "Blur 35 px",
    "combo_15_50": "Blur 15 +\ndark 50%",
    "combo_35_80": "Blur 35 +\ndark 80%",
    "low_contrast": "Low\ncontrast",
    "fog": "Fog",
    "occlusion_25": "Occlusion\n25%",
}


def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            row = dict(raw)
            for key in (
                "recall_at_1", "recall_at_5", "mean_separation_margin",
                "warm_mean_ms", "warm_p95_ms", "descriptor_dimension",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def grouped(rows):
    output = defaultdict(dict)
    for row in rows:
        output[row["model"]][row["condition"]] = row
    return output


def model_order(data):
    preferred = list(DISPLAY_NAMES)
    return [model for model in preferred if model in data] + sorted(
        set(data) - set(preferred)
    )


def aggregate(model_rows):
    values = list(model_rows.values())
    return {
        "recall": float(np.mean([row["recall_at_1"] for row in values])),
        "margin": float(np.mean([
            row["mean_separation_margin"] for row in values
        ])),
        "latency": float(np.mean([row["warm_mean_ms"] for row in values])),
        "p95": float(np.mean([row["warm_p95_ms"] for row in values])),
        "dimension": int(values[0]["descriptor_dimension"]),
    }


def style_axis(axis):
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_tradeoff(data, models, output):
    background = "#454D54"
    foreground = "#F4F4F4"
    yellow = "#F9C909"
    presentation_colours = {
        "places365_resnet50_embedding": "#9EA4A9",
        "eigenplaces_r18_512": yellow,
        "mixvpr_r50_512": "#C2C6C9",
        "salad": "#858B90",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans"],
        "font.size": 18,
        "text.color": foreground,
        "axes.labelcolor": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "svg.fonttype": "none",
    })
    fig, axis = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    fig.patch.set_facecolor(background)
    axis.set_facecolor(background)
    offsets = {
        "places365_resnet50_embedding": (14, 8),
        "eigenplaces_r18_512": (14, -58),
        "mixvpr_r50_512": (18, 12),
        "salad": (-18, -52),
    }
    alignments = {
        "salad": "right",
    }
    for model in models:
        stats = aggregate(data[model])
        axis.scatter(
            stats["latency"], stats["margin"], s=430,
            color=presentation_colours.get(model, "#B8BDC2"),
            edgecolor=foreground, linewidth=1.6,
            zorder=3,
        )
        method_name = DISPLAY_NAMES.get(model, model)
        if model == "eigenplaces_r18_512":
            method_name = "EigenPlaces (HuMem-VPR)"
        axis.annotate(
            f'{method_name}\nR@1 = {stats["recall"]:.2f}',
            (stats["latency"], stats["margin"]),
            xytext=offsets.get(model, (6, 6)),
            textcoords="offset points", fontsize=19,
            ha=alignments.get(model, "left"),
            color=yellow if model == "eigenplaces_r18_512" else foreground,
            weight="bold",
        )
    axis.set_xlabel("Mean warmed inference latency (ms)", fontsize=22,
                    weight="bold", labelpad=12)
    axis.set_ylabel("Positive–hardest-negative separation", fontsize=22,
                    weight="bold", labelpad=12)
    axis.set_xlim(3.1, 13.7)
    axis.set_ylim(0.02, 0.41)
    axis.tick_params(labelsize=18)
    axis.grid(True, color="white", alpha=0.10, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "accuracy_latency_tradeoff.png", dpi=300,
                bbox_inches="tight", facecolor=background)
    fig.savefig(output / "accuracy_latency_tradeoff.svg",
                bbox_inches="tight", facecolor=background)
    plt.close(fig)


def plot_conditions(data, models, output, metric, ylabel, filename, ylim=None):
    conditions = [condition for condition in CONDITION_ORDER
                  if any(condition in data[model] for model in models)]
    x_values = np.arange(len(conditions))
    width = 0.8 / len(models)
    fig, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    for index, model in enumerate(models):
        values = [data[model][condition][metric] for condition in conditions]
        axis.bar(
            x_values + (index - (len(models) - 1) / 2) * width,
            values, width=width, label=DISPLAY_NAMES.get(model, model),
            color=COLORS.get(model), alpha=0.9,
        )
    axis.set_xticks(x_values, [CONDITION_NAMES.get(item, item)
                              for item in conditions], fontsize=8.5)
    axis.set_ylabel(ylabel)
    if ylim:
        axis.set_ylim(*ylim)
    axis.legend(ncols=len(models), frameon=False, loc="upper center")
    style_axis(axis)
    fig.savefig(output / f"{filename}.png", dpi=220)
    fig.savefig(output / f"{filename}.pdf")
    plt.close(fig)


def plot_latency(data, models, output):
    stats = [aggregate(data[model]) for model in models]
    x_values = np.arange(len(models))
    fig, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = axis.bar(
        x_values, [item["latency"] for item in stats],
        color=[COLORS.get(model) for model in models], width=0.65,
    )
    axis.errorbar(
        x_values, [item["latency"] for item in stats],
        yerr=[item["p95"] - item["latency"] for item in stats],
        fmt="none", ecolor="black", capsize=4, linewidth=1,
    )
    axis.bar_label(bars, labels=[f'{item["latency"]:.2f} ms' for item in stats],
                   padding=3, fontsize=9)
    axis.set_xticks(x_values, [DISPLAY_NAMES.get(model, model)
                              for model in models])
    axis.set_ylabel("Warmed TensorRT inference latency (ms) ↓")
    axis.set_title("Descriptor extraction latency (mean; p95 whisker)")
    style_axis(axis)
    fig.savefig(output / "descriptor_latency.png", dpi=220)
    fig.savefig(output / "descriptor_latency.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = grouped(load_rows(args.summary))
    models = model_order(data)
    plot_tradeoff(data, models, args.output)
    plot_conditions(
        data, models, args.output, "mean_separation_margin",
        "Positive–hardest-negative separation ↑", "robustness_by_condition",
    )
    plot_conditions(
        data, models, args.output, "recall_at_1", "Recall@1 ↑",
        "recall_at_1_by_condition", (0.0, 1.08),
    )
    plot_latency(data, models, args.output)
    print(f"Wrote descriptor plots to {args.output}")


if __name__ == "__main__":
    main()
