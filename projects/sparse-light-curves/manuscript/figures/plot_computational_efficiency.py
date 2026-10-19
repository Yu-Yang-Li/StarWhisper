#!/usr/bin/env python3
"""Regenerate fig_computational_efficiency.pdf from deploy benchmark CSV."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "train_models/results/inference_time_benchmark_deploy.csv"
OUT = Path(__file__).resolve().parent

N_TEST = 257_742

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    }
)

COLORS = {
    "feat": "#4472C4",
    "infer": "#70AD47",
}

# (exp_id, y-axis label lines, panel-b bar color)
MODELS: list[tuple[str, list[str], str]] = [
    ("xgb_1117", ["XGBoost-Full", "(CPU)"], "#8DB4E2"),
    ("xgb_1121", ["XGBoost-Reduced", "(CPU)"], "#5B9BD5"),
    ("tf_feat_scratch", ["Feature-based", "Transformer", "(scratch, GPU)"], "#70AD47"),
    ("e2e_tf_matched_ft", ["E2E Transformer", "(pre-trained, GPU)"], "#D97272"),
    ("e2e_tf_small_ft", ["E2E Transformer Small", "(ft, GPU)"], "#D98AAB"),
    ("rnn_e2e_varlen", ["LSTM", "(GPU)"], "#E0AD72"),
]

BAR_HEIGHT = 0.78


def style_axes(ax: plt.Axes) -> None:
    """Keep only black x/y axis spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", colors="black", width=1.0, length=4)
    ax.grid(False)


def y_labels() -> list[str]:
    return ["\n".join(lines) for _, lines, _ in MODELS]


def load_rows() -> dict[str, dict[str, float]]:
    df = pd.read_csv(CSV_PATH)
    df = df[df["状态"] == "OK"].copy()
    rows: dict[str, dict[str, float]] = {}
    for exp_id, _, _ in MODELS:
        rec = df.loc[df["exp_id"] == exp_id].iloc[0]
        feat = float(rec["特征提取_秒"])
        infer = float(rec["推理_秒"])
        total = float(rec["总耗时_秒"])
        rows[exp_id] = {
            "feat_per_10k": feat / N_TEST * 10_000,
            "infer_per_10k": infer / N_TEST * 10_000,
            "total_per_10k": total / N_TEST * 10_000,
            "throughput": N_TEST / total,
        }
    return rows


def fmt_time(v: float) -> str:
    return f"{v:.1f}"


def fmt_throughput(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.0f}"


def apply_y_labels(ax: plt.Axes, y: np.ndarray, labels: list[str]) -> None:
    ax.set_yticks(y)
    ax.set_yticklabels(labels, linespacing=0.88)
    ax.tick_params(axis="y", pad=8)
    for tick in ax.get_yticklabels():
        tick.set_ha("right")


def main() -> None:
    data = load_rows()
    labels = y_labels()

    feat_vals = [data[eid]["feat_per_10k"] for eid, _, _ in MODELS]
    infer_vals = [data[eid]["infer_per_10k"] for eid, _, _ in MODELS]
    throughput = [data[eid]["throughput"] for eid, _, _ in MODELS]
    bar_colors = [color for _, _, color in MODELS]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    y = np.arange(len(MODELS))

    ax_a.barh(
        y,
        feat_vals,
        BAR_HEIGHT,
        label="Feature Extraction",
        color=COLORS["feat"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax_a.barh(
        y,
        infer_vals,
        BAR_HEIGHT,
        left=feat_vals,
        label="Model Inference",
        color=COLORS["infer"],
        edgecolor="white",
        linewidth=0.5,
    )

    max_total = max(f + i for f, i in zip(feat_vals, infer_vals))
    for i, (exp_id, _, _) in enumerate(MODELS):
        total = data[exp_id]["total_per_10k"]
        ax_a.text(
            total + max_total * 0.02,
            y[i],
            fmt_time(total),
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax_a.set_xlabel("Processing Time per 10k Samples (s)")
    apply_y_labels(ax_a, y, labels)
    ax_a.set_xlim(0, max(max_total * 1.18, 42))
    ax_a.invert_yaxis()
    ax_a.set_title("(a) Inference Time Breakdown", fontweight="bold", pad=6)
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.95)
    style_axes(ax_a)

    bars = ax_b.barh(
        y,
        throughput,
        BAR_HEIGHT,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_b.set_xlabel("Throughput (samples/s)")
    apply_y_labels(ax_b, y, labels)
    ax_b.set_xlim(0, 17_500)
    ax_b.invert_yaxis()
    ax_b.set_title("(b) Processing Throughput", fontweight="bold", pad=6)

    for bar, thr in zip(bars, throughput):
        ax_b.text(
            thr + 350,
            bar.get_y() + bar.get_height() / 2,
            fmt_throughput(thr),
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    style_axes(ax_b)

    fig.subplots_adjust(left=0.28, right=0.98, wspace=0.40, top=0.90, bottom=0.14)
    out_pdf = OUT / "fig_computational_efficiency.pdf"
    out_png = OUT / "fig_computational_efficiency.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
