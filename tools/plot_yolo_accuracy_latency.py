#!/usr/bin/env python3
"""Plot the frozen HuMemSLAM YOLO accuracy/latency selection evidence."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation_assets/model_selection_accuracy_latency/object_yolo"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = ["COCO\nPyTorch", "Mapillary\nPyTorch", "Mapillary\nTensorRT FP16"]
    median_latency = [13.01, 13.30, 4.84]
    p95_latency = [15.37, 15.14, 6.02]



    accuracy_names = ["Mapillary\nPyTorch", "Mapillary\nTensorRT FP16"]
    mask_map50 = [0.1366, 0.1393]
    mask_map5095 = [0.0572, 0.0582]

    background, foreground = "#454D54", "#F4F4F4"
    greys = ["#9EA4A9", "#C2C6C9", "#F9C909"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 18, "text.color": foreground,
                         "axes.labelcolor": foreground, "xtick.color": foreground,
                         "ytick.color": foreground, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(background)
    fig.subplots_adjust(left=.07, right=.985, top=.90, bottom=.24, wspace=.16)
    for axis in axes:
        axis.set_facecolor(background)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(labelsize=18)
        axis.grid(axis="y", alpha=.12, color="white", linewidth=.8)

    x = np.arange(len(names))
    width = 0.34
    for index, colour in enumerate(greys):
        axes[0].bar(x[index] - width / 2, median_latency[index], width, color=colour)
        axes[0].bar(x[index] + width / 2, p95_latency[index], width, color=colour,
                    alpha=.50, hatch="//", edgecolor=foreground, linewidth=.8)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Detector latency (ms)", fontsize=22, weight="bold")
    axes[0].text(.5, 1.02, "Latency", transform=axes[0].transAxes, ha="center",
                 fontsize=22, weight="bold", color=foreground)
    for offset, values in ((-width / 2, median_latency), (width / 2, p95_latency)):
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + .38, f"{value:.2f}", ha="center",
                         fontsize=18, weight="bold", color=foreground)

    x2 = np.arange(len(accuracy_names))
    accuracy_colours = ["#B8BDC2", "#F9C909"]
    for index, colour in enumerate(accuracy_colours):
        axes[1].bar(x2[index] - width / 2, mask_map50[index], width, color=colour)
        axes[1].bar(x2[index] + width / 2, mask_map5095[index], width, color=colour,
                    alpha=.50, hatch="//", edgecolor=foreground, linewidth=.8)
    axes[1].set_xticks(x2, accuracy_names)
    axes[1].set_ylabel("Mask mAP", fontsize=22, weight="bold")
    axes[1].set_ylim(0, .18)
    axes[1].text(.5, 1.02, "Segmentation accuracy", transform=axes[1].transAxes,
                 ha="center", fontsize=22, weight="bold", color=foreground)
    for offset, values in ((-width / 2, mask_map50), (width / 2, mask_map5095)):
        for index, value in enumerate(values):
            axes[1].text(index + offset, value + .005, f"{value:.4f}", ha="center",
                         fontsize=18, weight="bold", color=foreground)

    legend = [Patch(facecolor=foreground, label="Median / mAP50"),
              Patch(facecolor=foreground, alpha=.50, hatch="//", label="P95 / mAP50–95")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               fontsize=18, labelcolor=foreground, bbox_to_anchor=(.5, .035),
               columnspacing=2.2, handlelength=2.0)
    fig.savefig(OUTPUT / "yolo_accuracy_latency_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "yolo_accuracy_latency_comparison.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
