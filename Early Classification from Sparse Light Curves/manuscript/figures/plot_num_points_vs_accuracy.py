#!/usr/bin/env python3
"""Regenerate fig_num_points_vs_accuracy.pdf from experiment CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "train_models"
OUT = Path(__file__).resolve().parent

C = {
    "xgb_full": "#E66101",
    "xgb_reduced": "#FDB863",
    "e2e_ft": "#5E3C99",
    "lstm": "#4DAF4A",
    "gray_ref": "#999999",
}
BLUE = "#2171B5"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> None:
    xgb_full = pd.read_csv(BASE / "xgboost_full/results/num_points_vs_accuracy.csv")
    xgb_reduced = pd.read_csv(
        BASE / "xgboost_reduced/results/less_feature_num_points_vs_accuracy.csv"
    )
    e2e_ft = pd.read_csv(
        BASE / "e2e_transformer_varlen_matched_finetune/results/num_points_vs_accuracy.csv"
    )
    lstm = pd.read_csv(BASE / "lstm_varlen/results/num_points_vs_accuracy.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    ax = axes[0]
    ax.plot(
        xgb_full["num_points"],
        xgb_full["accuracy"] * 100,
        "o-",
        color=C["xgb_full"],
        label="XGBoost-Full (57 feat., with LS)",
        markersize=2.5,
        linewidth=1.0,
    )
    ax.plot(
        xgb_reduced["num_points"],
        xgb_reduced["accuracy"] * 100,
        "s-",
        color=C["xgb_reduced"],
        label="XGBoost-Reduced (39 feat., no LS)",
        markersize=2.5,
        linewidth=1.0,
    )
    ax.axvspan(3, 10, alpha=0.08, color=BLUE)
    ax.axvline(x=10, color=BLUE, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(6.5, 30, "Sparse regime\n(<10 obs.)", fontsize=7, color=BLUE, ha="center", alpha=0.85)
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) XGBoost-Full vs. XGBoost-Reduced", fontweight="bold")
    ax.legend(fontsize=6.5, framealpha=0.9, loc="lower right")
    ax.set_xlim(2, 31)
    ax.set_ylim(25, 102)
    ax.grid(alpha=0.3, linewidth=0.5)

    ax = axes[1]
    ax.plot(
        e2e_ft["num_points"],
        e2e_ft["accuracy"] * 100,
        "D-",
        color=C["e2e_ft"],
        label="E2E Transformer (pre-trained)",
        markersize=2.5,
        linewidth=1.2,
        zorder=3,
    )
    ax.plot(
        xgb_reduced["num_points"],
        xgb_reduced["accuracy"] * 100,
        "s-",
        color=C["xgb_reduced"],
        label="XGBoost-Reduced",
        markersize=2.5,
        linewidth=1.0,
        zorder=2,
    )
    ax.plot(
        lstm["num_points"],
        lstm["accuracy"] * 100,
        "^-",
        color=C["lstm"],
        label="LSTM",
        markersize=2.5,
        linewidth=1.0,
        zorder=2,
    )
    ax.axhline(y=90, color=C["gray_ref"], linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(y=99, color=C["gray_ref"], linestyle="--", linewidth=0.7, alpha=0.7)
    ax.text(29, 90.8, "90%", fontsize=7, color=C["gray_ref"], va="bottom")
    ax.text(29, 99.8, "99%", fontsize=7, color=C["gray_ref"], va="bottom")
    ax.annotate(
        ">90% at 7 obs",
        xy=(7, 92.03),
        xytext=(10.5, 87),
        fontsize=7,
        color=BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85,
            edgecolor=BLUE,
            linewidth=0.5,
        ),
    )
    ax.annotate(
        ">99% at 11 obs",
        xy=(11, 99.11),
        xytext=(16.5, 96.5),
        fontsize=7,
        color=BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85,
            edgecolor=BLUE,
            linewidth=0.5,
        ),
    )
    ax.annotate(
        "XGBoost\nbetter at\n3–4 obs",
        xy=(3.5, 64),
        xytext=(9, 54),
        fontsize=6.5,
        color=BLUE,
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7, alpha=0.7),
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            alpha=0.85,
            edgecolor=BLUE,
            linewidth=0.4,
        ),
    )
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Representative Models Comparison", fontweight="bold")
    ax.legend(fontsize=6.5, framealpha=0.9, loc="lower right")
    ax.set_xlim(2, 31)
    ax.set_ylim(25, 102)
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_pdf = OUT / "fig_num_points_vs_accuracy.pdf"
    out_png = OUT / "fig_num_points_vs_accuracy.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
