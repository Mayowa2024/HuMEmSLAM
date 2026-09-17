#!/usr/bin/env python3
"""Plot the frozen Business School Fall OCR accuracy/latency comparison."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation_assets/model_selection_accuracy_latency/text_ocr"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    methods = ["Paddle GPU", "TensorRT OCR"]
    recall_1 = [0.44, 0.44]
    mean_latency = [40.32, 32.26]

    background = "#454D54"
    foreground = "#F4F4F4"
    grey = "#B8BDC2"
    yellow = "#F9C909"

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 18, "text.color": foreground,
                         "axes.labelcolor": foreground, "xtick.color": foreground,
                         "ytick.color": foreground, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    fig.patch.set_facecolor(background)
    axis.set_facecolor(background)
    colours = [grey, yellow]
    axis.scatter(mean_latency, recall_1, s=520, c=colours, edgecolors=foreground,
                 linewidths=1.8, zorder=3)
    axis.annotate("Paddle GPU\n40.32 ms, R@1 = 0.44", (mean_latency[0], recall_1[0]),
                  xytext=(42.75, .462), textcoords="data", ha="right", fontsize=20,
                  color=foreground, weight="bold")
    axis.annotate("TensorRT OCR\n32.26 ms, R@1 = 0.44", (mean_latency[1], recall_1[1]),
                  xytext=(29.55, .462), textcoords="data", ha="left", fontsize=20,
                  color=yellow, weight="bold")
    axis.annotate("8.06 ms lower latency", xy=(32.55, .44), xytext=(39.95, .44),
                  arrowprops={"arrowstyle": "->", "color": yellow, "lw": 2.8},
                  ha="right", va="bottom", fontsize=20, color=yellow, weight="bold")
    axis.set_xlim(29, 44)
    axis.set_ylim(.38, .50)
    axis.set_xlabel("Mean end-to-end HuMem-VPR latency (ms)", fontsize=22, weight="bold")
    axis.set_ylabel("Recall@1", fontsize=22, weight="bold")
    axis.tick_params(labelsize=18)
    axis.grid(alpha=.12, color="white", linewidth=.8)
    for spine in axis.spines.values():
        spine.set_visible(False)

    fig.savefig(OUTPUT / "ocr_frozen_retrieval_accuracy_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "ocr_frozen_retrieval_accuracy_latency.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
