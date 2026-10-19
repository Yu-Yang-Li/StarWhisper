#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总各模型在统一 75/10/15 测试集上的分类指标。

指标来源：各实验目录下 test_metrics.json / metrics.txt
  - 准确率 (Accuracy)
  - 宏平均 F1 (Macro-F1)

主对比图仅包含 varlen/1121 池上的最终部署阶段（3–30 变长/微调/端到端）。
50obs 预训练结果单独输出到 comparison_pretrain.png。

实验清单见 experiment_registry.py。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

from experiment_registry import collect_metrics_table

BASE = Path("/root/shared-nvme/train_models")
OUT = BASE / "results"
INFER_CSV = OUT / "inference_time_benchmark.csv"
DEPLOY_CSV = OUT / "inference_time_benchmark_deploy.csv"
OUT.mkdir(parents=True, exist_ok=True)

PRETRAIN_STAGES = {"50点预训练"}

_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
_CJK_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
]


def setup_cjk_font() -> str:
    """Pick the first available CJK font for matplotlib."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name

    for path in _CJK_FONT_PATHS:
        if not path.is_file():
            continue
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return name

    for f in font_manager.fontManager.ttflist:
        if "Noto Sans CJK" in f.name:
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return f.name

    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans (fallback, no CJK font found)"


setup_cjk_font()


def is_pretrain_row(row: pd.Series) -> bool:
    return row["阶段"] in PRETRAIN_STAGES or row["数据池"] == "50obs"


def is_main_deploy_row(row: pd.Series) -> bool:
    return row["状态"] == "OK" and not is_pretrain_row(row)


def plot_metrics_bars(
    df: pd.DataFrame,
    png_path: Path,
    suptitle: str,
    group_png_path: Path | None = None,
) -> None:
    if df.empty:
        return

    sub = df.sort_values("宏平均F1", ascending=False)
    x = range(len(sub))
    labels = [f"{r['模型']}\n({r['阶段']})" for _, r in sub.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x, sub["宏平均F1"], color="steelblue", alpha=0.85)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("测试集 Macro-F1")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, sub["准确率"], color="coral", alpha=0.85)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("测试集 Accuracy")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    if group_png_path is None:
        return

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    groups_order = list(dict.fromkeys(sub["实验组"]))
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(groups_order), 1)))
    g2c = {g: palette[i] for i, g in enumerate(groups_order)}
    ax2.bar(
        x,
        sub["宏平均F1"],
        color=[g2c[g] for g in sub["实验组"]],
        alpha=0.85,
    )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(sub["模型"], rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Macro-F1")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("按实验组 — 测试集 Macro-F1")
    legend_handles = [
        Patch(facecolor=g2c[g], edgecolor="grey", linewidth=0.4, alpha=0.85, label=g)
        for g in groups_order
    ]
    ax2.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(group_png_path, dpi=200, bbox_inches="tight")
    plt.close()


def merge_inference_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """合并 GPU 部署向推理 benchmark（优先 deploy 表，否则回退完整 CSV）。"""
    bench_path = DEPLOY_CSV if DEPLOY_CSV.is_file() else INFER_CSV
    if not bench_path.is_file():
        for col in (
            "推理_秒",
            "特征提取_秒",
            "总耗时_秒",
            "每万样本_秒",
            "峰值显存_MB",
            "权重_MB",
            "推理测速备注",
        ):
            df[col] = None
        return df

    bench = pd.read_csv(bench_path)
    ok = bench[bench["状态"].astype(str).str.startswith("OK")].copy()
    if ok.empty:
        return df

    prefer = ok.copy()
    if "device" in prefer.columns and (prefer["device"] == "gpu").any():
        prefer = prefer[prefer["device"] == "gpu"]
    prefer = prefer.sort_values("每万样本_秒").drop_duplicates("exp_id", keep="first")
    cols = {
        "推理_秒": "推理_秒",
        "特征提取_秒": "特征提取_秒",
        "总耗时_秒": "总耗时_秒",
        "每万样本_秒": "每万样本_秒",
        "峰值显存_MB": "峰值显存_MB",
        "权重_MB": "权重_MB",
        "测速备注": "推理测速备注",
    }
    pick = ["exp_id"] + [c for c in cols if c in prefer.columns]
    merged = df.merge(prefer[pick], on="exp_id", how="left")
    rename = {k: v for k, v in cols.items() if k in merged.columns}
    return merged.rename(columns=rename)


def main() -> None:
    rows = collect_metrics_table(include_resources=True)
    df = pd.DataFrame(rows)
    df = merge_inference_benchmark(df)

    df.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")

    ok = df[df["状态"] == "OK"].copy()
    main_df = ok[ok.apply(is_main_deploy_row, axis=1)].copy()
    pretrain_df = ok[ok.apply(is_pretrain_row, axis=1)].copy()

    main_df.sort_values("宏平均F1", ascending=False).to_csv(
        OUT / "model_comparison_main.csv", index=False, encoding="utf-8-sig"
    )
    pretrain_df.sort_values("宏平均F1", ascending=False).to_csv(
        OUT / "model_comparison_pretrain.csv", index=False, encoding="utf-8-sig"
    )
    ok.sort_values("宏平均F1", ascending=False).to_csv(
        OUT / "model_comparison_available.csv", index=False, encoding="utf-8-sig"
    )
    resource_cols = [
        "exp_id",
        "模型",
        "实验组",
        "阶段",
        "宏平均F1",
        "权重文件_MB",
        "参数量_M",
        "计划epoch",
        "实际epoch",
        "最佳epoch",
        "训练墙钟_小时",
        "训练备注",
        "每万样本_秒",
        "峰值显存_MB",
        "推理测速备注",
    ]
    resource_cols = [c for c in resource_cols if c in df.columns]
    ok.sort_values("宏平均F1", ascending=False)[resource_cols].to_csv(
        OUT / "model_comparison_resources.csv", index=False, encoding="utf-8-sig"
    )

    print("=" * 80)
    print("统一划分 (75/10/15, random_state=42) — 全部实验")
    print("=" * 80)
    show_cols = ["模型", "实验组", "阶段", "数据池", "准确率", "宏平均F1", "n_test", "状态"]
    resource_show = ["权重文件_MB", "参数量_M", "训练墙钟_小时", "每万样本_秒"]
    resource_show = [c for c in resource_show if c in df.columns]
    print(df[show_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("主对比（varlen/1121，3–30 部署场景，不含 50obs 预训练）")
    print("=" * 80)
    if main_df.empty:
        print("(无可用指标)")
    else:
        print(main_df[show_cols + resource_show].sort_values("宏平均F1", ascending=False).to_string(index=False))

    print("\n" + "=" * 80)
    print("50obs 预训练（中间阶段，单独参考）")
    print("=" * 80)
    if pretrain_df.empty:
        print("(无可用指标)")
    else:
        print(pretrain_df[show_cols].sort_values("宏平均F1", ascending=False).to_string(index=False))

    missing = df[df["状态"] != "OK"]["模型"].tolist()
    if missing:
        print(f"\n缺失指标（尚未训练或路径不对）: {', '.join(missing)}")

    if not main_df.empty:
        plot_metrics_bars(
            main_df,
            OUT / "comparison.png",
            "模型对比 — varlen/1121 部署场景（75/10/15 test）",
            OUT / "comparison_by_group.png",
        )

    if not pretrain_df.empty:
        plot_metrics_bars(
            pretrain_df,
            OUT / "comparison_pretrain.png",
            "50obs 预训练 checkpoint（50obs 池 test，不与 varlen 主图混比）",
        )

    print(f"\n已保存: {OUT / 'model_comparison_resources.csv'}")
    print(f"已保存: {OUT / 'model_comparison.csv'}")
    print(f"已保存: {OUT / 'model_comparison_main.csv'}")
    print(f"已保存: {OUT / 'model_comparison_pretrain.csv'}")
    if not main_df.empty:
        print(f"已保存: {OUT / 'comparison.png'}")
        print(f"已保存: {OUT / 'comparison_by_group.png'}")
    if not pretrain_df.empty:
        print(f"已保存: {OUT / 'comparison_pretrain.png'}")


if __name__ == "__main__":
    main()
