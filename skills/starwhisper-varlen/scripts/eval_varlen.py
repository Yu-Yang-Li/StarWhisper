"""Compare the published sparse-LC varlen benchmark. Stdlib only. Does not train."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
TABLE = SKILL / "references" / "published_metrics.csv"
CONTRACT = json.loads((SKILL / "references" / "contract.json").read_text(encoding="utf-8"))
MAIN_POOL = "varlen"


def load_rows() -> list[dict]:
    rows = []
    with TABLE.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["accuracy"] = float(row["accuracy"])
            row["macro_f1"] = float(row["macro_f1"])
            row["random_state"] = int(row["random_state"])
            row["n_test"] = int(row["n_test"]) if row.get("n_test") else None
            rows.append(row)
    return rows


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["exp_id"]: row for row in rows}


def pool_rows(rows: list[dict], pool: str) -> list[dict]:
    return [row for row in rows if row["pool"] == pool]


def emit(payload: dict, as_json: bool, table: list[list[str]] | None = None, header: list[str] | None = None) -> int:
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    if header and table is not None:
        widths = [max(len(header[i]), *(len(r[i]) for r in table) if table else 0) for i in range(len(header))]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        for row in table:
            print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    for note in payload.get("must_state", []):
        print(f"! {note}")
    return 0


def caveats() -> list[str]:
    return [
        "compare only inside one pool; 50obs accuracy is not a varlen 3-30 result",
        "a test-set metric is not an explosion time or a discovery",
        "do not quote StarWhisper LC Kepler/K2 ~90% as this benchmark",
    ]


def cmd_contract(args, rows):
    payload = {
        "main_setting": MAIN_POOL,
        "contract": CONTRACT,
        "must_state": caveats(),
    }
    if not args.json:
        v = CONTRACT["varlen"]
        print(f"main setting: varlen, {v['n_obs'][0]}-{v['n_obs'][1]} observations, 7 merged classes")
        print(f"split {v['split']}, random_state={v['random_state']}, n_test={v['n_test']}")
        print("merged labels:", ", ".join(CONTRACT["merged_categories"]))
    return emit(payload, args.json)


def cmd_table(args, rows):
    selected = pool_rows(rows, args.pool) if args.pool else rows
    selected = sorted(selected, key=lambda r: r["macro_f1"], reverse=True)
    payload = {
        "pool": args.pool or "all",
        "n": len(selected),
        "rows": selected,
        "must_state": caveats(),
    }
    header = ["exp_id", "pool", "group", "acc", "macro_f1", "n_test"]
    table = [
        [
            r["exp_id"],
            r["pool"],
            r["group"],
            f"{r['accuracy']:.4f}",
            f"{r['macro_f1']:.4f}",
            str(r["n_test"] or "-"),
        ]
        for r in selected
    ]
    return emit(payload, args.json, table, header)


def cmd_best(args, rows):
    pool = args.pool or MAIN_POOL
    selected = sorted(pool_rows(rows, pool), key=lambda r: r["macro_f1"], reverse=True)
    if not selected:
        raise SystemExit(f"no rows in pool {pool}")
    winner = selected[0]
    payload = {
        "pool": pool,
        "best": winner,
        "ranked": [{"exp_id": r["exp_id"], "macro_f1": r["macro_f1"]} for r in selected],
        "must_state": caveats(),
    }
    if not args.json:
        print(f"best in {pool}: {winner['exp_id']}  macro_f1={winner['macro_f1']:.4f}  acc={winner['accuracy']:.4f}")
    return emit(payload, args.json)


def cmd_compare(args, rows):
    index = by_id(rows)
    if args.a not in index or args.b not in index:
        raise SystemExit(f"unknown exp_id; have {sorted(index)}")
    a, b = index[args.a], index[args.b]
    same_pool = a["pool"] == b["pool"]
    payload = {
        "a": a,
        "b": b,
        "same_pool": same_pool,
        "delta": {
            "accuracy": round(a["accuracy"] - b["accuracy"], 4),
            "macro_f1": round(a["macro_f1"] - b["macro_f1"], 4),
        },
        "comparable": same_pool,
        "must_state": caveats(),
    }
    if not same_pool:
        payload["must_state"] = [
            f"{a['exp_id']} is pool={a['pool']}, {b['exp_id']} is pool={b['pool']}; do not treat the delta as a method gain"
        ] + payload["must_state"]
    if not args.json:
        mark = "OK" if same_pool else "NOT COMPARABLE"
        print(f"{a['exp_id']} vs {b['exp_id']}  [{mark}]")
        print(f"  d_acc={payload['delta']['accuracy']:+.4f}  d_macro_f1={payload['delta']['macro_f1']:+.4f}")
    emit(payload, args.json)
    return 0 if same_pool else 1


def cmd_labels(args, rows):
    merge = CONTRACT["merge"]
    raw = [x.strip() for x in args.raw.split(",") if x.strip()] if args.raw else CONTRACT["raw_categories"]
    mapped, unknown = [], []
    for name in raw:
        if name in merge:
            mapped.append({"raw": name, "merged": merge[name]})
        else:
            unknown.append(name)
    payload = {
        "mapped": mapped,
        "unknown": unknown,
        "merged_categories": CONTRACT["merged_categories"],
        "must_state": ["unknown raw labels must be added to MERGE_MAPPING before training, not silently dropped"],
    }
    code = 1 if unknown else 0
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return code
    for row in mapped:
        print(f"{row['raw']:8} -> {row['merged']}")
    if unknown:
        print("unknown:", ", ".join(unknown))
    for note in payload["must_state"]:
        print(f"! {note}")
    return code


def read_user_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    return rows, fields


def cmd_check(args, _rows):
    data, fields = read_user_csv(args.csv)
    lo, hi = CONTRACT["varlen"]["n_obs"]
    merge = CONTRACT["merge"]
    merged = set(CONTRACT["merged_categories"])
    findings = []
    nobs_key = args.nobs_col
    label_key = args.label_col
    if nobs_key not in fields:
        findings.append({"level": "error", "message": f"no {nobs_key!r} column; found {fields}"})
    if label_key not in fields:
        findings.append({"level": "error", "message": f"no {label_key!r} column; found {fields}"})
    outside, unknown, empty = 0, [], 0
    for i, row in enumerate(data, start=2):
        if nobs_key in fields:
            raw = row.get(nobs_key, "")
            try:
                n = int(float(raw))
            except ValueError:
                findings.append({"level": "error", "message": f"{nobs_key} {raw!r} is not a number", "where": f"line {i}"})
                continue
            if not lo <= n <= hi:
                outside += 1
        if label_key in fields:
            lab = row.get(label_key, "")
            if not lab:
                empty += 1
            elif lab not in merged and lab not in merge:
                unknown.append(lab)
    if outside:
        findings.append({"level": "error", "message": f"{outside} rows have {nobs_key} outside {lo}-{hi}"})
    if empty:
        findings.append({"level": "error", "message": f"{empty} rows have empty {label_key}"})
    unknown = sorted(set(unknown))
    if unknown:
        findings.append({"level": "error", "message": f"unknown labels {unknown}; expected merged {sorted(merged)} or raw {CONTRACT['raw_categories']}"})
    payload = {
        "csv": str(args.csv),
        "n_rows": len(data),
        "findings": findings,
        "errors": sum(1 for f in findings if f["level"] == "error"),
        "must_state": caveats(),
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 1 if payload["errors"] else 0
    print(f"{payload['n_rows']} rows")
    for f in findings:
        where = f" ({f['where']})" if f.get("where") else ""
        print(f"  [{f['level'].upper()}] {f['message']}{where}")
    if not findings:
        print("contract OK for varlen labels and n_obs")
    for note in payload["must_state"]:
        print(f"! {note}")
    return 1 if payload["errors"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("contract", help="print the varlen 3-30 / 7-class contract")
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser("table", help="rank published configs")
    p.add_argument("--pool", choices=["varlen", "50obs", "1121"])
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("best", help="best published config inside one pool")
    p.add_argument("--pool", default=MAIN_POOL, choices=["varlen", "50obs", "1121"])
    p.set_defaults(func=cmd_best)

    p = sub.add_parser("compare", help="delta between two exp_id values")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("labels", help="map raw ZTF/ATLAS folder names to 7 merged classes")
    p.add_argument("--raw", help="comma-separated raw labels; default: print the full map")
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("check", help="lint a user CSV against the varlen contract")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--nobs-col", default="n_obs")
    p.add_argument("--label-col", default="label")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args, load_rows())


if __name__ == "__main__":
    raise SystemExit(main())
