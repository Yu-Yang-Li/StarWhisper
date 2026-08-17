#!/usr/bin/env python3
"""Regression tests for run_stat_tests.py: numbers cross-checked against
scipy/statsmodels called directly, plus CLI end-to-end smoke on a temp CSV.
Run: python tests/regression.py  (all lines should be PASS)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stat_tests.py"
PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS  {name}")
    else:
        FAILED += 1
        print(f"FAIL  {name}  {detail}")


def run_cli(*args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        return None, proc.stderr
    return json.loads(proc.stdout), proc.stderr


def make_csv(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


def main():
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    a = rng.normal(10, 2, 40)
    b = rng.normal(11.5, 2, 45)
    c = rng.normal(12.5, 2, 38)

    rows = ["value,arm"]
    for name, data in (("A", a), ("B", b), ("C", c)):
        rows += [f"{v:.6f},{name}" for v in data]
    csv3 = make_csv("\n".join(rows))
    rows2 = ["value,arm"] + [f"{v:.6f},A" for v in a] + [f"{v:.6f},B" for v in b]
    csv2 = make_csv("\n".join(rows2))

    # --- describe ---
    out, err = run_cli("describe", "--csv", csv2, "--value", "value", "--group", "arm")
    check("describe runs", out is not None, err)
    if out:
        check("describe n matches", out["groups"]["A"]["n"] == 40 and out["groups"]["B"]["n"] == 45)
        check("describe mean matches numpy",
              abs(out["groups"]["A"]["mean"] - float(np.mean(np.round(a, 6)))) < 1e-4)

    # --- welch t vs scipy ---
    out, err = run_cli("ttest", "--csv", csv2, "--value", "value", "--group", "arm")
    check("ttest runs", out is not None, err)
    if out:
        ref_t, ref_p = stats.ttest_ind(np.round(a, 6), np.round(b, 6), equal_var=False)
        check("welch t statistic matches scipy", abs(out["statistic"] - round(float(ref_t), 4)) < 1e-6)
        check("welch p matches scipy", abs(out["p_value"] - round(float(ref_p), 6)) < 1e-6)
        check("effect size present", out["effect_size"]["cohen_d"] is not None)
        check("hedges g smaller than d in magnitude",
              abs(out["effect_size"]["hedges_g"]) < abs(out["effect_size"]["cohen_d"]))
        ci = out["ci95_mean_diff"]
        check("ci brackets mean diff", ci["low"] < ci["mean_diff"] < ci["high"])
        check("assumption checks attached", len(out["assumptions"]) == 3)

    # --- mann-whitney vs scipy ---
    out, err = run_cli("mannwhitney", "--csv", csv2, "--value", "value", "--group", "arm")
    check("mannwhitney runs", out is not None, err)
    if out:
        ref_u, ref_p = stats.mannwhitneyu(np.round(a, 6), np.round(b, 6), alternative="two-sided")
        check("U matches scipy", abs(out["statistic"] - round(float(ref_u), 4)) < 1e-6)
        rb = 1 - 2 * float(ref_u) / (len(a) * len(b))
        check("rank-biserial matches formula", abs(out["effect_size"]["rank_biserial_r"] - round(rb, 4)) < 1e-6)

    # --- anova vs scipy + tukey ---
    out, err = run_cli("anova", "--csv", csv3, "--value", "value", "--group", "arm", "--posthoc")
    check("anova runs", out is not None, err)
    if out:
        ref_f, ref_p = stats.f_oneway(np.round(a, 6), np.round(b, 6), np.round(c, 6))
        check("F matches scipy", abs(out["statistic_F"] - round(float(ref_f), 4)) < 1e-6)
        check("eta squared in (0,1)", 0 < out["effect_size"]["eta_squared"] < 1)
        check("tukey posthoc has 3 pairs", len(out["posthoc_tukey"]) == 3)

    # --- kruskal vs scipy ---
    out, err = run_cli("kruskal", "--csv", csv3, "--value", "value", "--group", "arm")
    check("kruskal runs", out is not None, err)
    if out:
        ref_h, _ = stats.kruskal(np.round(a, 6), np.round(b, 6), np.round(c, 6))
        check("H matches scipy", abs(out["statistic_H"] - round(float(ref_h), 4)) < 1e-6)

    # --- paired t vs scipy ---
    pre = rng.normal(50, 5, 30)
    post = pre + rng.normal(2, 1.5, 30)
    csvp = make_csv("pre,post\n" + "\n".join(f"{x:.6f},{y:.6f}" for x, y in zip(pre, post)))
    out, err = run_cli("paired", "--csv", csvp, "--before", "pre", "--after", "post")
    check("paired runs", out is not None, err)
    if out:
        ref_t, ref_p = stats.ttest_rel(np.round(pre, 6), np.round(post, 6))
        check("paired t matches scipy", abs(out["statistic"] - round(float(ref_t), 4)) < 1e-6)
        check("paired n correct", out["n_pairs"] == 30)

    # --- chi2 vs scipy (with fisher on 2x2) ---
    chi_rows = ["exposure,outcome"]
    chi_rows += ["yes,case"] * 30 + ["yes,control"] * 20 + ["no,case"] * 15 + ["no,control"] * 35
    csvc = make_csv("\n".join(chi_rows))
    out, err = run_cli("chi2", "--csv", csvc, "--row", "exposure", "--col", "outcome")
    check("chi2 runs", out is not None, err)
    if out:
        import pandas as pd

        frame = pd.read_csv(csvc)
        table = pd.crosstab(frame["exposure"], frame["outcome"])
        ref_chi2, ref_p, _, _ = stats.chi2_contingency(table)
        check("chi2 matches scipy", abs(out["statistic_chi2"] - round(float(ref_chi2), 4)) < 1e-6)
        check("fisher attached for 2x2", "fisher_exact" in out)
        check("cramers v in (0,1)", 0 < out["effect_size"]["cramers_v"] < 1)

    # --- corr vs scipy ---
    x = rng.normal(0, 1, 60)
    y = 0.6 * x + rng.normal(0, 0.8, 60)
    csvr = make_csv("x,y\n" + "\n".join(f"{i:.6f},{j:.6f}" for i, j in zip(x, y)))
    out, err = run_cli("corr", "--csv", csvr, "--x", "x", "--y", "y")
    check("corr runs", out is not None, err)
    if out:
        ref_r, ref_p = stats.pearsonr(np.round(x, 6), np.round(y, 6))
        check("pearson r matches scipy", abs(out["r"] - round(float(ref_r), 4)) < 1e-6)
        check("r ci brackets r", out["ci95_r"]["low"] < out["r"] < out["ci95_r"]["high"])
        check("corr boundary mentions causality", "因果" in out["interpretation_boundary"])

    # --- multiple comparison correction vs statsmodels ---
    out, err = run_cli("correct", "--pvalues", "0.01,0.04,0.03,0.005", "--method", "holm")
    check("correct runs", out is not None, err)
    if out:
        from statsmodels.stats.multitest import multipletests

        _, ref_adj, _, _ = multipletests([0.01, 0.04, 0.03, 0.005], alpha=0.05, method="holm")
        check("holm adjusted matches statsmodels",
              all(abs(o - round(float(r), 6)) < 1e-6 for o, r in zip(out["adjusted_p"], ref_adj)))

    # --- guardrails ---
    out, err = run_cli("ttest", "--csv", csv3, "--value", "value", "--group", "arm")
    check("ttest rejects 3 groups", out is None and "恰好两组" in err, err[:120] if err else "")

    print(f"\n==== {PASSED} passed, {FAILED} failed ====")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
