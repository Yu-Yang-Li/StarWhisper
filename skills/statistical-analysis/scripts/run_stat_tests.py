#!/usr/bin/env python3
"""Post-collection confirmatory statistical tests (deterministic, tool-backed).

Executes the analysis plan locked before data collection: descriptive stats,
assumption checks, hypothesis tests with effect sizes and CIs, and multiple
comparison correction. Wraps scipy/statsmodels so numbers are reproducible
instead of hand-estimated by the model.

Usage:
  python run_stat_tests.py describe    --csv data.csv --value y --group arm
  python run_stat_tests.py ttest       --csv data.csv --value y --group arm [--equal-var]
  python run_stat_tests.py paired      --csv data.csv --before pre --after post
  python run_stat_tests.py mannwhitney --csv data.csv --value y --group arm
  python run_stat_tests.py anova       --csv data.csv --value y --group arm [--posthoc]
  python run_stat_tests.py kruskal     --csv data.csv --value y --group arm
  python run_stat_tests.py chi2        --csv data.csv --row exposure --col outcome
  python run_stat_tests.py corr        --csv data.csv --x a --y b [--method pearson|spearman]
  python run_stat_tests.py correct     --pvalues 0.01,0.03,0.2 --method holm

Requires: pip install pandas scipy statsmodels
"""
from __future__ import annotations

import argparse
import json
import math
import sys

try:  # Windows 默认 gbk 代码页在重定向时会因中文输出崩溃，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _need():
    try:
        import pandas  # noqa: F401
        import scipy.stats  # noqa: F401
        import statsmodels.stats.multitest  # noqa: F401
    except Exception:
        sys.exit("需要 pandas / scipy / statsmodels：pip install pandas scipy statsmodels")


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))


def load_column(csv_path, column):
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if column not in frame.columns:
        sys.exit(f"CSV 中没有列 {column!r}；现有列：{list(frame.columns)}")
    return frame


def numeric_series(frame, column):
    import pandas as pd

    series = pd.to_numeric(frame[column], errors="coerce")
    dropped = int(series.isna().sum() - frame[column].isna().sum())
    if dropped > 0:
        print(f"# 注意：列 {column!r} 有 {dropped} 个非数值条目被剔除", file=sys.stderr)
    return series


def split_groups(frame, value, group):
    values = numeric_series(frame, value)
    grouped = {}
    for name, idx in frame.groupby(group, dropna=True).groups.items():
        data = values.loc[idx].dropna().to_numpy()
        if len(data):
            grouped[str(name)] = data
    if len(grouped) < 2:
        sys.exit(f"分组列 {group!r} 下有效组不足两个：{list(grouped)}")
    return grouped


def assumption_checks(groups):
    """Shapiro per group (3<=n<=5000) + Brown-Forsythe variance check."""
    from scipy import stats

    checks = []
    for name, data in groups.items():
        if 3 <= len(data) <= 5000:
            stat, p = stats.shapiro(data)
            checks.append({
                "name": f"shapiro_normality[{name}]",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "pass": bool(p > 0.05),
            })
        else:
            checks.append({
                "name": f"shapiro_normality[{name}]",
                "note": f"n={len(data)} 超出 Shapiro 适用范围，跳过",
            })
    if len(groups) >= 2:
        stat, p = stats.levene(*groups.values(), center="median")
        checks.append({
            "name": "levene_equal_variance(Brown-Forsythe)",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "pass": bool(p > 0.05),
        })
    return checks


def cohen_d_independent(a, b):
    import numpy as np

    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return None, None
    d = (np.mean(a) - np.mean(b)) / pooled
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)  # Hedges 小样本校正
    return float(d), float(d * j)


def welch_ci(a, b, alpha=0.05):
    import numpy as np
    from scipy import stats

    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        return None
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = float(np.mean(a) - np.mean(b))
    return {"mean_diff": round(diff, 6), "low": round(diff - t_crit * se, 6), "high": round(diff + t_crit * se, 6)}


def boundary_line(p, kind):
    """把统计结论的证据边界说清楚，不替用户把话说满。"""
    if kind == "corr":
        return "相关不代表因果；是否有实际意义要看效应量与领域背景。"
    if p is None:
        return "无法计算 p 值；检查数据量与取值是否退化。"
    if p < 0.05:
        return "组间差异统计上显著；是否重要要结合效应量和最小实际意义效应（SESOI）判断。"
    return "未检测到显著差异；这不等于证明无差异，可能是功效不足。"


def cmd_describe(args):
    frame = load_column(args.csv, args.value)
    groups = split_groups(frame, args.value, args.group)
    import numpy as np

    out = {}
    for name, data in groups.items():
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        out[name] = {
            "n": int(len(data)),
            "mean": round(float(np.mean(data)), 6),
            "sd": round(float(np.std(data, ddof=1)), 6) if len(data) > 1 else None,
            "median": round(float(med), 6),
            "iqr": [round(float(q1), 6), round(float(q3), 6)],
            "min": round(float(np.min(data)), 6),
            "max": round(float(np.max(data)), 6),
        }
    emit({"test": "describe", "value": args.value, "group": args.group, "groups": out,
          "missing_in_value_column": int(frame[args.value].isna().sum())})


def two_groups(args):
    frame = load_column(args.csv, args.value)
    groups = split_groups(frame, args.value, args.group)
    if len(groups) != 2:
        sys.exit(f"该检验需要恰好两组，当前 {len(groups)} 组：{list(groups)}；多组请用 anova/kruskal")
    return groups


def cmd_ttest(args):
    from scipy import stats

    groups = two_groups(args)
    (name_a, a), (name_b, b) = groups.items()
    equal_var = bool(args.equal_var)
    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    d, g = cohen_d_independent(a, b)
    emit({
        "test": "student_t" if equal_var else "welch_t",
        "groups": {name_a: len(a), name_b: len(b)},
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": {"cohen_d": round(d, 4) if d is not None else None,
                        "hedges_g": round(g, 4) if g is not None else None},
        "ci95_mean_diff": welch_ci(a, b),
        "assumptions": assumption_checks(groups),
        "interpretation_boundary": boundary_line(p, "diff"),
    })


def cmd_paired(args):
    import numpy as np
    from scipy import stats

    frame = load_column(args.csv, args.before)
    if args.after not in frame.columns:
        sys.exit(f"CSV 中没有列 {args.after!r}")
    before = numeric_series(frame, args.before)
    after = numeric_series(frame, args.after)
    mask = before.notna() & after.notna()
    a, b = before[mask].to_numpy(), after[mask].to_numpy()
    if len(a) < 3:
        sys.exit("配对样本不足 3 对")
    stat, p = stats.ttest_rel(a, b)
    diff = a - b
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd else None
    se = sd / math.sqrt(len(diff)) if sd else None
    t_crit = stats.t.ppf(0.975, len(diff) - 1)
    emit({
        "test": "paired_t",
        "n_pairs": int(len(diff)),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": {"cohen_dz": round(d, 4) if d is not None else None},
        "ci95_mean_diff": {"mean_diff": round(float(np.mean(diff)), 6),
                           "low": round(float(np.mean(diff) - t_crit * se), 6),
                           "high": round(float(np.mean(diff) + t_crit * se), 6)} if se else None,
        "assumptions": assumption_checks({"diff": diff}),
        "interpretation_boundary": boundary_line(p, "diff"),
    })


def cmd_mannwhitney(args):
    from scipy import stats

    groups = two_groups(args)
    (name_a, a), (name_b, b) = groups.items()
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 1 - 2 * float(stat) / (len(a) * len(b))
    emit({
        "test": "mann_whitney_u",
        "groups": {name_a: len(a), name_b: len(b)},
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": {"rank_biserial_r": round(rank_biserial, 4)},
        "interpretation_boundary": boundary_line(p, "diff"),
    })


def cmd_anova(args):
    import numpy as np
    from scipy import stats

    frame = load_column(args.csv, args.value)
    groups = split_groups(frame, args.value, args.group)
    if len(groups) < 3:
        print("# 注意：只有两组时 anova 等价于 t 检验，通常直接用 ttest", file=sys.stderr)
    stat, p = stats.f_oneway(*groups.values())
    all_data = np.concatenate(list(groups.values()))
    grand = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values())
    ss_total = float(np.sum((all_data - grand) ** 2))
    eta_sq = ss_between / ss_total if ss_total else None
    result = {
        "test": "one_way_anova",
        "groups": {k: len(v) for k, v in groups.items()},
        "statistic_F": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": {"eta_squared": round(float(eta_sq), 4) if eta_sq is not None else None},
        "assumptions": assumption_checks(groups),
        "interpretation_boundary": boundary_line(p, "diff"),
    }
    if args.posthoc:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        values = np.concatenate(list(groups.values()))
        labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
        tukey = pairwise_tukeyhsd(values, labels)
        result["posthoc_tukey"] = [
            {"group_a": str(r[0]), "group_b": str(r[1]), "mean_diff": round(float(r[2]), 6),
             "p_adj": round(float(r[3]), 6), "reject": bool(r[6])}
            for r in tukey.summary().data[1:]
        ]
    emit(result)


def cmd_kruskal(args):
    from scipy import stats

    frame = load_column(args.csv, args.value)
    groups = split_groups(frame, args.value, args.group)
    stat, p = stats.kruskal(*groups.values())
    n = sum(len(g) for g in groups.values())
    k = len(groups)
    epsilon_sq = (stat - k + 1) / (n - k) if n > k else None
    emit({
        "test": "kruskal_wallis",
        "groups": {k2: len(v) for k2, v in groups.items()},
        "statistic_H": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": {"epsilon_squared": round(float(epsilon_sq), 4) if epsilon_sq is not None else None},
        "interpretation_boundary": boundary_line(p, "diff"),
    })


def cmd_chi2(args):
    import pandas as pd
    from scipy import stats

    frame = load_column(args.csv, args.row)
    if args.col not in frame.columns:
        sys.exit(f"CSV 中没有列 {args.col!r}")
    table = pd.crosstab(frame[args.row], frame[args.col])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    min_dim = min(table.shape) - 1
    cramers_v = math.sqrt(chi2 / (n * min_dim)) if n and min_dim else None
    warnings = []
    if (expected < 5).any():
        warnings.append("有期望频数 < 5 的单元格；2x2 表建议改用 Fisher 精确检验")
    result = {
        "test": "chi2_contingency",
        "table_shape": list(table.shape),
        "n": n,
        "statistic_chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": round(float(p), 6),
        "effect_size": {"cramers_v": round(float(cramers_v), 4) if cramers_v is not None else None},
        "warnings": warnings,
        "interpretation_boundary": boundary_line(p, "diff"),
    }
    if table.shape == (2, 2):
        odds, fisher_p = stats.fisher_exact(table)
        result["fisher_exact"] = {"odds_ratio": round(float(odds), 4), "p_value": round(float(fisher_p), 6)}
    emit(result)


def cmd_corr(args):
    from scipy import stats

    frame = load_column(args.csv, args.x)
    if args.y not in frame.columns:
        sys.exit(f"CSV 中没有列 {args.y!r}")
    x = numeric_series(frame, args.x)
    y = numeric_series(frame, args.y)
    mask = x.notna() & y.notna()
    xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
    if len(xv) < 4:
        sys.exit("有效配对数据不足 4 条")
    if args.method == "spearman":
        r, p = stats.spearmanr(xv, yv)
        ci = None
    else:
        r, p = stats.pearsonr(xv, yv)
        z = math.atanh(max(min(r, 0.999999), -0.999999))
        se = 1 / math.sqrt(len(xv) - 3)
        ci = {"low": round(math.tanh(z - 1.959964 * se), 4), "high": round(math.tanh(z + 1.959964 * se), 4)}
    emit({
        "test": f"{args.method}_correlation",
        "n": int(len(xv)),
        "r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "ci95_r": ci,
        "interpretation_boundary": boundary_line(p, "corr"),
    })


def cmd_correct(args):
    from statsmodels.stats.multitest import multipletests

    pvals = [float(v) for v in args.pvalues.split(",") if v.strip()]
    if not pvals:
        sys.exit("--pvalues 为空")
    reject, adjusted, _, _ = multipletests(pvals, alpha=args.alpha, method=args.method)
    emit({
        "test": "multiple_comparison_correction",
        "method": args.method,
        "alpha": args.alpha,
        "input_p": pvals,
        "adjusted_p": [round(float(v), 6) for v in adjusted],
        "reject": [bool(v) for v in reject],
        "note": "所有做过的检验都应进入校正，不能只报显著的那部分。",
    })


def main():
    _need()
    parser = argparse.ArgumentParser(description="采集后验证性统计分析（scipy/statsmodels 对拍）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_vg(p):
        p.add_argument("--csv", required=True)
        p.add_argument("--value", required=True)
        p.add_argument("--group", required=True)

    add_vg(sub.add_parser("describe", help="分组描述统计"))
    t = sub.add_parser("ttest", help="独立两组 t 检验（默认 Welch）")
    add_vg(t)
    t.add_argument("--equal-var", action="store_true", help="使用 Student t（假定等方差）")
    pr = sub.add_parser("paired", help="配对 t 检验")
    pr.add_argument("--csv", required=True)
    pr.add_argument("--before", required=True)
    pr.add_argument("--after", required=True)
    add_vg(sub.add_parser("mannwhitney", help="Mann-Whitney U（两组非参数）"))
    an = sub.add_parser("anova", help="单因素 ANOVA")
    add_vg(an)
    an.add_argument("--posthoc", action="store_true", help="附 Tukey HSD 事后两两比较")
    add_vg(sub.add_parser("kruskal", help="Kruskal-Wallis（多组非参数）"))
    c2 = sub.add_parser("chi2", help="列联表卡方检验")
    c2.add_argument("--csv", required=True)
    c2.add_argument("--row", required=True)
    c2.add_argument("--col", required=True)
    co = sub.add_parser("corr", help="相关分析")
    co.add_argument("--csv", required=True)
    co.add_argument("--x", required=True)
    co.add_argument("--y", required=True)
    co.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    cr = sub.add_parser("correct", help="多重比较校正")
    cr.add_argument("--pvalues", required=True, help="逗号分隔的 p 值列表")
    cr.add_argument("--method", choices=["bonferroni", "holm", "fdr_bh"], default="holm")
    cr.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    {
        "describe": cmd_describe,
        "ttest": cmd_ttest,
        "paired": cmd_paired,
        "mannwhitney": cmd_mannwhitney,
        "anova": cmd_anova,
        "kruskal": cmd_kruskal,
        "chi2": cmd_chi2,
        "corr": cmd_corr,
        "correct": cmd_correct,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
