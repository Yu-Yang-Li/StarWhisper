#!/usr/bin/env python3
"""Regenerate confusion-matrix figures with True label / Predicted label axis titles."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

OUT = Path(__file__).resolve().parent
JSON_PATH = OUT / "confusion_matrices.json"

CLASS_LABELS = ["Active", "CV", "Eclip.", "LPV", "Puls.", "RR", "SN"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

CMAP = plt.get_cmap("Blues")
NORM = Normalize(vmin=0.0, vmax=1.0)


def load_matrices() -> dict[str, np.ndarray]:
    with JSON_PATH.open() as f:
        raw = json.load(f)
    return {name: np.asarray(matrix, dtype=float) for name, matrix in raw.items()}


def plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str | None = None,
) -> plt.cm.ScalarMappable:
    n = matrix.shape[0]
    im = ax.imshow(matrix, cmap=CMAP, norm=NORM, aspect="equal", origin="upper")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CLASS_LABELS, rotation=0, ha="center")
    ax.set_yticklabels(CLASS_LABELS, rotation=90, va="center", ha="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label", rotation=90, labelpad=8)
    if title:
        ax.set_title(title, fontweight="bold", pad=8)

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            color = "white" if value > 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)

    return im


def fig_confusion_matrices(matrices: dict[str, np.ndarray]) -> None:
    specs = [
        ("XGBoost-Reduced", "(a) XGBoost-Reduced"),
        ("E2E Transformer (pre-trained)", "(b) E2E Transformer (pre-trained)"),
        ("LSTM", "(c) LSTM"),
        ("E2E Transformer (no pre-training)", "(d) E2E Transformer (no pre-training)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    im = None
    for ax, (key, title) in zip(axes.flat, specs):
        im = plot_heatmap(ax, matrices[key], title=title)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recall")

    fig.subplots_adjust(left=0.10, right=0.9, top=0.96, bottom=0.06, wspace=0.18, hspace=0.18)
    out = OUT / "fig_confusion_matrices.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_appendix_confusion_xgb_full(matrices: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = plot_heatmap(
        ax,
        matrices["XGBoost-Full"],
        title="XGBoost-Full (57 features, with LS)",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall")
    fig.tight_layout()
    out = OUT / "fig_appendix_confusion_xgb_full.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_appendix_confusion_transformers(matrices: dict[str, np.ndarray]) -> None:
    specs = [
        ("Feature-based Transformer (scratch)", "(a) Feature-based Transformer (scratch)"),
        ("Feature-based Transformer (fine-tuned)", "(b) Feature-based Transformer (fine-tuned)"),
        ("E2E Transformer Small (ft)", "(c) E2E Transformer Small (ft)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])

    im = None
    for ax, (key, title) in zip(axes, specs):
        im = plot_heatmap(ax, matrices[key], title=title)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recall")

    fig.subplots_adjust(left=0.12, right=0.9, top=0.88, bottom=0.12, wspace=0.22)
    out = OUT / "fig_appendix_confusion_transformers.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    matrices = load_matrices()
    fig_confusion_matrices(matrices)
    fig_appendix_confusion_xgb_full(matrices)
    fig_appendix_confusion_transformers(matrices)


if __name__ == "__main__":
    main()
