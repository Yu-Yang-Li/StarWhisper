#!/usr/bin/env python3
"""experiment-design 回归测试套件（无第三方框架，纯 subprocess + 断言）。
用法:
  python tests/regression.py            # 默认测同级 ../scripts
  python tests/regression.py <scripts_dir>
针对交付件真实脚本运行；power.py 的数值结果与 statsmodels 独立对拍。
需要 statsmodels（与 power.py 相同依赖）。
"""
import subprocess, sys, os, re, math, csv, tempfile
from collections import Counter

SCRIPTS = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
PY = sys.executable
PASS = 0
FAIL = []


def run(script, *args, want_code=None):
    r = subprocess.run([PY, os.path.join(SCRIPTS, script), *args],
                       capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout or "") + (r.stderr or "")
    if want_code is not None and r.returncode != want_code:
        raise AssertionError(f"exit={r.returncode} want={want_code} :: {out[:200]}")
    return r.returncode, out


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(f"{name}: {detail}")


def no_traceback(out):
    return "Traceback (most recent call last)" not in out


# ---------- power.py 数值对拍 ----------
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower, GofChisquarePower

for d in [0.2, 0.35, 0.5, 0.8, 1.2]:
    for pw in [0.8, 0.9]:
        _, out = run("power.py", "ttest", "--effect", str(d), "--power", str(pw), want_code=0)
        got = int(re.search(r"组1 N = (\d+)", out).group(1))
        exp = math.ceil(TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=pw,
                                                    ratio=1.0, alternative="two-sided"))
        check(f"ttest d={d} pw={pw}", got == exp, f"got {got} exp {exp}")

for f_ in [0.1, 0.25, 0.4]:
    for k in [2, 3, 4, 5]:
        _, out = run("power.py", "anova", "--effect", str(f_), "--k", str(k), want_code=0)
        per = int(re.search(r"每组 N = (\d+)", out).group(1))
        tot = int(re.search(r"总 N = (\d+)", out).group(1))
        check(f"anova total f={f_} k={k}", per * k == tot, f"{per}*{k}!={tot}")
        ach = FTestAnovaPower().solve_power(effect_size=f_, nobs=tot, alpha=0.05, k_groups=k)
        check(f"anova>=0.8 f={f_} k={k}", ach >= 0.8, f"power={ach:.4f}")
        if per > 1:
            below = FTestAnovaPower().solve_power(effect_size=f_, nobs=(per - 1) * k,
                                                  alpha=0.05, k_groups=k)
            check(f"anova minimal f={f_} k={k}", below < 0.8, f"below={below:.4f}")

for w in [0.1, 0.3, 0.5]:
    for b in [2, 4, 6]:
        _, out = run("power.py", "chisq", "--effect", str(w), "--nbins", str(b), want_code=0)
        got = int(re.search(r"总 N = (\d+)", out).group(1))
        exp = math.ceil(GofChisquarePower().solve_power(effect_size=w, alpha=0.05, power=0.8, n_bins=b))
        check(f"chisq w={w} b={b}", got == exp, f"got {got} exp {exp}")

# ttest ratio: group2 = ratio*group1
_, out = run("power.py", "ttest", "--effect", "0.5", "--power", "0.8", "--ratio", "2", want_code=0)
n1 = int(re.search(r"组1 N = (\d+)", out).group(1))
n2 = int(re.search(r"组2 N = (\d+)", out).group(1))
check("ttest ratio 2", n2 == n1 * 2, f"{n1},{n2}")

# ---------- power.py 边界必须干净失败 ----------
for args in [["ttest", "--effect", "0", "--power", "0.8"],
             ["anova", "--effect", "0.25", "--k", "1"],
             ["ttest", "--effect", "0.5", "--alpha", "0"],
             ["ttest", "--effect", "0.5", "--power", "1.5"],
             ["ttest", "--effect", "0.5", "--ratio", "0"],
             ["power", "--test", "ttest", "--effect", "0.5", "--n", "1"],
             ["power", "--test", "chisq", "--effect", "0.3", "--n", "100"],
             ["mde", "--test", "anova", "--n", "50"]]:
    code, out = run("power.py", *args)
    check(f"power reject {args[0]}:{args[1:]}", code != 0 and no_traceback(out), out[:150])

# 负效应量取绝对值且成功
code, out = run("power.py", "ttest", "--effect", "-0.5", "--power", "0.8")
check("power neg->abs", code == 0 and "组1 N = 64" in out, out[:150])

# ---------- randomization.py ----------
# 区组平衡（每区组 + 整体）
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "b.csv")
    run("randomization.py", "block", "--n", "60", "--arms", "A,B,C", "--seed", "5", "--out", p, want_code=0)
    arms = [r[1] for r in list(csv.reader(open(p, encoding="utf-8-sig")))[1:]]
    check("block overall balance", Counter(arms) == Counter({"A": 20, "B": 20, "C": 20}), str(Counter(arms)))
    ok = all(set(Counter(arms[i:i+6]).values()) == {2} for i in range(0, 60, 6))
    check("block per-block balance", ok)

# 可复现
_, a = run("randomization.py", "block", "--n", "30", "--arms", "A,B", "--seed", "9", want_code=0)
_, b = run("randomization.py", "block", "--n", "30", "--arms", "A,B", "--seed", "9", want_code=0)
check("rand reproducible", a == b)

# 分层比例精确
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "s.csv")
    run("randomization.py", "stratified", "--strata", "s1:40", "--arms", "A,B", "--ratio", "3,1",
        "--seed", "2", "--out", p, want_code=0)
    c = Counter(r["arm"] for r in csv.DictReader(open(p, encoding="utf-8-sig")))
    check("stratified 3:1", c == Counter({"A": 30, "B": 10}), str(c))

# --out 自动建目录 + 中文
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "新 目录", "方案.csv")
    code, out = run("randomization.py", "block", "--n", "4", "--arms", "处理,对照", "--seed", "1", "--out", p)
    check("rand out auto-mkdir zh", code == 0 and os.path.exists(p), out[:150])

# 边界干净失败
for args in [["stratified", "--strata", "s1:30", "--arms", "A,B,C", "--ratio", "2,1"],
             ["stratified", "--strata", "s130", "--arms", "A,B"],
             ["simple", "--n", "5", "--arms", ",,"],
             ["cluster", "--clusters", "c1", "--arms", "A,B,C"],
             ["simple", "--n", "0", "--arms", "A,B"]]:
    code, out = run("randomization.py", *args)
    check(f"rand reject {args[0]}", code != 0 and no_traceback(out), out[:150])

# ---------- doe_designs.py ----------
# 计数正确
_, out = run("doe_designs.py", "full2", "--factor", "a:0,1", "--factor", "b:0,1", "--factor", "c:0,1", want_code=0)
check("full2 2^3=8", "8 次运行" in out, out[:80])
_, out = run("doe_designs.py", "full", "--factor", "t:20,40,60", "--factor", "c:A,B", want_code=0)
check("full 3x2=6", "6 次运行" in out, out[:80])

# LHS 空间填充 + 可复现
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "l.csv")
    run("doe_designs.py", "lhs", "--factor", "x:0,100", "--factor", "y:0,50", "--n", "12",
        "--seed", "3", "--out", p, want_code=0)
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for dim, lo, hi in [("x", 0, 100), ("y", 0, 50)]:
        strata = set(int((float(r[dim]) - lo) / (hi - lo) * 12) for r in rows)
        check(f"lhs spacefill {dim}", strata == set(range(12)), str(sorted(strata)))

# 值含冒号（分类）
_, out = run("doe_designs.py", "full", "--factor", "time:9:00,10:00", want_code=0)
check("doe colon value", "2 次运行" in out, out[:80])

# 边界干净失败
for args in [["lhs", "--factor", "cat:A,B", "--n", "5"],
             ["full", "--factor", "temp"],
             ["lhs", "--factor", "t:0,1", "--n", "1"]]:
    code, out = run("doe_designs.py", *args)
    check(f"doe reject {args}", code != 0 and no_traceback(out), out[:150])

# ---------- 汇总 ----------
print(f"\n==== {PASS} passed, {len(FAIL)} failed ====")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
