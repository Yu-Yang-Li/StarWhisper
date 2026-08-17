#!/usr/bin/env python3
"""DOE 实验设计矩阵生成（真实因子单位、随机化运行顺序、seeded；仅用标准库）。

零依赖实现全析因/两水平析因/拉丁超立方设计。运行顺序默认随机化，防因子与时间/漂移混杂。

用法（因子写成 名:low,high，可多次）：
  python doe_designs.py full2 --factor temp:20,60 --factor conc:1,10 --factor pH:6,8 --seed 42
  python doe_designs.py full  --factor temp:20,40,60 --factor cat:A,B --seed 42
  python doe_designs.py lhs   --factor temp:20,60 --factor conc:1,10 --n 8 --seed 42
  加 --out design.csv 导出
"""
from __future__ import annotations
import argparse, csv, itertools, os, random, sys

try:  # Windows gbk 代码页在重定向/被捕获时会因中文崩溃，强制 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_factor(s):
    if ":" not in s:
        raise SystemExit(f"--factor 格式应为 名:v1,v2[,...]，无法解析 '{s}'")
    name, vals = s.split(":", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--factor 缺少因子名：'{s}'")
    parts = [p.strip() for p in vals.split(",") if p.strip()]
    if not parts:
        raise SystemExit(f"--factor '{name}' 至少需要 1 个水平值")
    # numeric if all parse as float
    try:
        parts_num = [float(p) for p in parts]
        return name.strip(), parts_num
    except ValueError:
        return name.strip(), parts


def full_factorial(factors):
    names = [f[0] for f in factors]
    levels = [f[1] for f in factors]
    rows = [dict(zip(names, combo)) for combo in itertools.product(*levels)]
    return names, rows


def two_level(factors):
    names = [f[0] for f in factors]
    lows_highs = [(f[1][0], f[1][-1]) for f in factors]
    rows = [dict(zip(names, combo)) for combo in itertools.product(*lows_highs)]
    return names, rows


def latin_hypercube(factors, n, seed):
    """连续因子的 LHS：每维分 n 层，各层取一点，跨维随机配对。"""
    rnd = random.Random(seed)
    names = [f[0] for f in factors]
    cols = {}
    for name, vals in factors:
        if not all(isinstance(v, float) for v in vals):
            raise SystemExit(f"拉丁超立方(lhs)只支持连续/数值因子，'{name}' 含非数值水平：{vals}")
        lo, hi = float(vals[0]), float(vals[-1])
        strata = list(range(n))
        rnd.shuffle(strata)
        cols[name] = [round(lo + (hi - lo) * (k + rnd.random()) / n, 4) for k in strata]
    rows = [{name: cols[name][i] for name in names} for i in range(n)]
    return names, rows


def main():
    ap = argparse.ArgumentParser(description="DOE 设计矩阵（seeded，运行顺序随机化）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("full", "full2", "lhs"):
        s = sub.add_parser(name)
        s.add_argument("--factor", action="append", required=True, dest="factors")
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--out", default="")
        if name == "lhs":
            s.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    factors = [parse_factor(f) for f in a.factors]

    if a.cmd == "full":
        names, rows = full_factorial(factors)
    elif a.cmd == "full2":
        names, rows = two_level(factors)
    elif a.cmd == "lhs":
        if a.n < 2:
            raise SystemExit(f"lhs 的 --n 必须 ≥ 2，收到 {a.n}")
        names, rows = latin_hypercube(factors, a.n, a.seed)

    rnd = random.Random(a.seed)
    order = list(range(len(rows)))
    rnd.shuffle(order)  # 随机化运行顺序

    print(f"# {a.cmd} 设计，{len(rows)} 次运行，seed={a.seed}（run_order 已随机化）")
    print("  run_order\t" + "\t".join(names))
    out_rows = []
    for run_pos, idx in enumerate(order, 1):
        r = rows[idx]
        out_rows.append({"run_order": run_pos, **r})
    out_rows.sort(key=lambda x: x["run_order"])
    for r in out_rows[:16]:
        print(f"  {r['run_order']}\t" + "\t".join(str(r[n]) for n in names))
    if len(out_rows) > 16:
        print(f"  ...（共 {len(out_rows)} 次运行）")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)   # 目标目录自动创建
        with open(a.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_order"] + names)
            w.writeheader(); w.writerows(out_rows)
        print(f"已导出 {a.out}")


if __name__ == "__main__":
    main()
