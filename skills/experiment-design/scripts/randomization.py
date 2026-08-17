#!/usr/bin/env python3
"""随机化/分配方案生成（确定性、seeded、可归档复现，仅用标准库）。

模型手工"随机分配"不可信也不可复现，这里给可 seed、可导出 CSV 的确定性方案，
覆盖简单/置换区组/分层区组/整群随机化。

用法：
  python randomization.py block --n 60 --arms treatment,control --seed 42
  python randomization.py stratified --strata siteA:30,siteB:30 --arms drug,placebo --ratio 2,1 --seed 42
  python randomization.py cluster --clusters c1,c2,c3,c4 --arms A,B --seed 42
  python randomization.py simple --n 40 --arms A,B,C --seed 42
  加 --out sched.csv 导出
"""
from __future__ import annotations
import argparse, csv, os, random, sys

try:  # Windows gbk 代码页在重定向/被捕获时会因中文崩溃，强制 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _arms(s): return [a.strip() for a in s.split(",") if a.strip()]


def simple(n, arms, seed):
    rnd = random.Random(seed)
    return [(i + 1, rnd.choice(arms)) for i in range(n)]


def block(n, arms, seed, block_mult=2):
    """置换区组：每个区组含各臂各 block_mult 个，组内打乱，保证全程平衡。"""
    rnd = random.Random(seed)
    block = []
    for a in arms:
        block += [a] * block_mult
    out, i = [], 0
    while len(out) < n:
        b = block[:]
        rnd.shuffle(b)
        for a in b:
            if len(out) >= n:
                break
            i += 1
            out.append((i, a))
    return out


def stratified(strata, arms, ratio, seed):
    rnd = random.Random(seed)
    unit = []
    for a, r in zip(arms, ratio):
        unit += [a] * r
    rows = []
    for stratum, cnt in strata.items():
        i = 0
        while i < cnt:
            b = unit[:]
            rnd.shuffle(b)
            for a in b:
                if i >= cnt:
                    break
                i += 1
                rows.append((stratum, i, a))
    return rows


def cluster(clusters, arms, seed):
    rnd = random.Random(seed)
    cl = clusters[:]
    rnd.shuffle(cl)
    return [(c, arms[i % len(arms)]) for i, c in enumerate(cl)]


def balance(rows, arm_idx=-1):
    from collections import Counter
    return dict(Counter(r[arm_idx] for r in rows))


def main():
    ap = argparse.ArgumentParser(description="seeded 随机化方案")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simple", "block"):
        s = sub.add_parser(name); s.add_argument("--n", type=int, required=True)
        s.add_argument("--arms", required=True); s.add_argument("--seed", type=int, default=42)
    st = sub.add_parser("stratified")
    st.add_argument("--strata", required=True, help="siteA:30,siteB:30")
    st.add_argument("--arms", required=True); st.add_argument("--ratio", default="")
    st.add_argument("--seed", type=int, default=42)
    cl = sub.add_parser("cluster")
    cl.add_argument("--clusters", required=True); cl.add_argument("--arms", required=True)
    cl.add_argument("--seed", type=int, default=42)
    for s in sub.choices.values():
        s.add_argument("--out", default="")
    a = ap.parse_args()

    if a.cmd == "simple":
        arms = _arms(a.arms)
        if not arms:
            sys.exit("--arms 至少需要 1 个非空臂")
        if a.n < 1:
            sys.exit(f"--n 必须 ≥ 1，收到 {a.n}")
        rows = simple(a.n, arms, a.seed); header = ["unit", "arm"]
    elif a.cmd == "block":
        arms = _arms(a.arms)
        if not arms:
            sys.exit("--arms 至少需要 1 个非空臂")
        if a.n < 1:
            sys.exit(f"--n 必须 ≥ 1，收到 {a.n}")
        rows = block(a.n, arms, a.seed); header = ["unit", "arm"]
    elif a.cmd == "stratified":
        strata = {}
        for part in a.strata.split(","):
            if ":" not in part:
                sys.exit(f"--strata 格式应为 名:数量[,名:数量]，无法解析片段 '{part}'")
            k, v = part.split(":", 1)
            try:
                strata[k.strip()] = int(v)
            except ValueError:
                sys.exit(f"--strata 中 '{k.strip()}' 的数量不是整数：'{v}'")
        arms = _arms(a.arms)
        if not arms:
            sys.exit("--arms 至少需要 1 个非空臂")
        ratio = [int(x) for x in a.ratio.split(",")] if a.ratio else [1] * len(arms)
        if len(ratio) != len(arms):
            sys.exit(f"--ratio 的项数（{len(ratio)}）必须与 --arms 的臂数（{len(arms)}）一致，"
                     f"否则会静默丢弃臂：arms={arms}, ratio={ratio}")
        rows = stratified(strata, arms, ratio, a.seed); header = ["stratum", "unit", "arm"]
    elif a.cmd == "cluster":
        clusters, arms = _arms(a.clusters), _arms(a.arms)
        if not arms:
            sys.exit("--arms 至少需要 1 个非空臂")
        if not clusters:
            sys.exit("--clusters 至少需要 1 个非空整群")
        if len(clusters) < len(arms):
            sys.exit(f"整群数（{len(clusters)}）少于臂数（{len(arms)}）：至少有一臂分不到整群")
        rows = cluster(clusters, arms, a.seed); header = ["cluster", "arm"]

    print(f"# {a.cmd} 随机化，seed={a.seed}，各臂计数：{balance(rows)}")
    for r in rows[:20]:
        print("  " + "\t".join(map(str, r)))
    if len(rows) > 20:
        print(f"  ...（共 {len(rows)} 行）")
    if a.out:
        parent = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(parent, exist_ok=True)   # 目标目录不存在时自动创建，避免 FileNotFoundError
        with open(a.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
        print(f"已导出 {a.out}")


if __name__ == "__main__":
    main()
