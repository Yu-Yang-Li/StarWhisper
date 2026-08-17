#!/usr/bin/env python3
"""A-priori sample-size / power calculator (deterministic, tool-backed).

Wraps statsmodels power classes so the experiment-design skill can return exact,
reproducible sample sizes instead of the model guessing numbers by hand.

Usage:
  python power.py ttest  --effect 0.5 --alpha 0.05 --power 0.80 [--ratio 1.0]
  python power.py anova  --effect 0.25 --alpha 0.05 --power 0.80 --k 4
  python power.py chisq  --effect 0.3 --alpha 0.05 --power 0.80 --nbins 4
  python power.py power   --test ttest --effect 0.5 --n 50 [--alpha 0.05]
  python power.py mde     --test ttest --n 64 --alpha 0.05 --power 0.80

Requires: pip install statsmodels
"""
from __future__ import annotations
import argparse, math, sys

try:  # Windows 默认 gbk 代码页在重定向/被捕获时会因中文和 ⚠/✓ 崩溃，强制 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _need():
    try:
        import statsmodels.stats.power as p  # noqa
    except Exception:
        sys.exit("需要 statsmodels：pip install statsmodels")


def _check_common(args):
    """采集前功效分析的参数边界校验，避免 statsmodels 返回 nan/数组导致的晦涩报错。"""
    a = getattr(args, "alpha", 0.05)
    if not (0 < a < 1):
        sys.exit(f"alpha 必须在 (0,1) 区间，收到 {a}")
    p = getattr(args, "power", None)
    if p is not None and not (0 < p < 1):
        sys.exit(f"power 必须在 (0,1) 区间，收到 {p}")
    e = getattr(args, "effect", None)
    if e is not None and e == 0:
        sys.exit("效应量不能为 0（0 效应需要无穷样本）；请基于文献/预实验/SESOI 给一个非零效应")
    if e is not None and e < 0:
        # 双侧检验只看绝对值，负号多半是笔误，取绝对值并提示
        args.effect = abs(e)
        print(f"# 注意：效应量取绝对值 {args.effect}（双侧检验与符号无关）")


def solve_n(args):
    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower, GofChisquarePower
    _check_common(args)
    if args.cmd == "anova" and args.k < 2:
        sys.exit(f"ANOVA 至少需要 2 组，收到 k={args.k}")
    if args.cmd == "chisq" and args.nbins < 2:
        sys.exit(f"卡方检验至少需要 2 个类别，收到 nbins={args.nbins}")
    if args.cmd == "ttest" and args.ratio <= 0:
        sys.exit(f"ratio 必须 > 0，收到 {args.ratio}")
    if args.cmd == "ttest":
        n1 = math.ceil(TTestIndPower().solve_power(effect_size=args.effect, alpha=args.alpha,
                                                   power=args.power, ratio=args.ratio, alternative="two-sided"))
        n2 = math.ceil(n1 * args.ratio)          # 第二组 = ratio × 第一组
        print(f"两样本 t 检验：组1 N = {n1}，组2 N = {n2}，总 N = {n1 + n2}")
        print(f"  参数 d={args.effect}, alpha={args.alpha}, power={args.power}, ratio={args.ratio}")
    elif args.cmd == "anova":
        # statsmodels FTestAnovaPower.solve_power 返回的是【总】样本量，不是每组。
        n_total_f = FTestAnovaPower().solve_power(effect_size=args.effect, alpha=args.alpha,
                                                  power=args.power, k_groups=args.k)
        per = math.ceil(n_total_f / args.k)     # 每组向上取整，保证均衡设计不欠功效
        n_total = per * args.k
        print(f"单因素 ANOVA（{args.k} 组）：每组 N = {per}，总 N = {n_total}")
        print(f"  参数 f={args.effect}, alpha={args.alpha}, power={args.power}, k={args.k}")
    elif args.cmd == "chisq":
        n = GofChisquarePower().solve_power(effect_size=args.effect, alpha=args.alpha,
                                            power=args.power, n_bins=args.nbins)
        print(f"卡方检验：总 N = {math.ceil(n)}（w={args.effect}, bins={args.nbins}, alpha={args.alpha}, power={args.power}）")


def solve_power(args):
    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
    _check_common(args)
    if args.n < 2:
        sys.exit(f"样本量 N 必须 ≥ 2，收到 {args.n}")
    if args.test == "anova" and args.k < 2:
        sys.exit(f"ANOVA 至少需要 2 组，收到 k={args.k}")
    if args.test == "ttest":
        pw = TTestIndPower().solve_power(effect_size=args.effect, alpha=args.alpha,
                                         nobs1=args.n, ratio=1.0, alternative="two-sided")
    elif args.test == "anova":
        pw = FTestAnovaPower().solve_power(effect_size=args.effect, alpha=args.alpha,
                                           nobs=args.n, k_groups=args.k)
    else:
        sys.exit("power 仅支持 --test ttest|anova")
    print(f"给定 N={args.n} 时的功效 = {pw:.3f}"
          + ("  ⚠ 欠功效(<0.8)" if pw < 0.8 else "  ✓"))


def solve_mde(args):
    from statsmodels.stats.power import TTestIndPower
    if args.test != "ttest":
        sys.exit("mde 目前仅支持 --test ttest")
    _check_common(args)
    if args.n < 2:
        sys.exit(f"样本量 N 必须 ≥ 2，收到 {args.n}")
    d = TTestIndPower().solve_power(nobs1=args.n, alpha=args.alpha, power=args.power,
                                    ratio=1.0, alternative="two-sided")
    print(f"给定每组 N={args.n}、power={args.power} 时可检测的最小效应 Cohen's d = {d:.3f}")


def main():
    _need()
    ap = argparse.ArgumentParser(description="a-priori 功效/样本量计算")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ttest", "anova", "chisq"):
        s = sub.add_parser(name)
        s.add_argument("--effect", type=float, required=True)
        s.add_argument("--alpha", type=float, default=0.05)
        s.add_argument("--power", type=float, default=0.80)
        s.add_argument("--ratio", type=float, default=1.0)
        s.add_argument("--k", type=int, default=3)
        s.add_argument("--nbins", type=int, default=4)
    sp = sub.add_parser("power")
    sp.add_argument("--test", default="ttest")
    sp.add_argument("--effect", type=float, required=True)
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--k", type=int, default=3)
    sm = sub.add_parser("mde")
    sm.add_argument("--test", default="ttest")
    sm.add_argument("--n", type=int, required=True)
    sm.add_argument("--alpha", type=float, default=0.05)
    sm.add_argument("--power", type=float, default=0.80)
    args = ap.parse_args()
    if args.cmd in ("ttest", "anova", "chisq"):
        solve_n(args)
    elif args.cmd == "power":
        solve_power(args)
    elif args.cmd == "mde":
        solve_mde(args)


if __name__ == "__main__":
    main()
