#!/usr/bin/env python3
"""Heuristic visual-quality audit for reconstructed editable deck layouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rounded_box_signature(text: dict) -> tuple[float, float] | None:
    try:
        return round(float(text["w"]), 3), round(float(text["h"]), 3)
    except (KeyError, TypeError, ValueError):
        return None


def _role(text: dict) -> str:
    return str(text.get("role") or "").strip().lower()


def audit(layered_path: Path) -> dict:
    deck = _load_json(layered_path)
    issues: list[dict] = []
    slides_out: list[dict] = []

    for index, slide in enumerate(deck.get("slides") or [], start=1):
        sid = str(slide.get("slide_id") or index)
        texts = [item for item in (slide.get("texts") or []) if isinstance(item, dict)]
        roles = Counter(_role(item) for item in texts if _role(item))
        signatures = Counter(sig for sig in (_rounded_box_signature(item) for item in texts) if sig)
        max_repeated_shape = max(signatures.values(), default=0)
        role_names = set(roles)
        slide_issues: list[dict] = []
        score = 100

        if len(texts) > 48:
            issue = {
                "level": "warning",
                "slide_id": sid,
                "message": "very high editable text density",
                "details": {"text_count": len(texts)},
            }
            issues.append(issue)
            slide_issues.append(issue)
            score -= 12

        has_chart = any(role.startswith("chart") for role in role_names)
        has_table = any(role.startswith("table") for role in role_names)
        has_legend = any(role.startswith("legend") for role in role_names)
        if len(texts) >= 20 and not (has_chart or has_table or has_legend):
            issue = {
                "level": "warning",
                "slide_id": sid,
                "message": "dense slide lacks chart, table, or legend semantics",
                "details": {"roles": sorted(role_names)},
            }
            issues.append(issue)
            slide_issues.append(issue)
            score -= 18

        if max_repeated_shape >= 10 and not has_table:
            issue = {
                "level": "warning",
                "slide_id": sid,
                "message": "many editable text boxes share identical dimensions",
                "details": {"max_repeated_box_shape": max_repeated_shape},
            }
            issues.append(issue)
            slide_issues.append(issue)
            score -= 10

        if has_chart and not has_legend:
            issue = {
                "level": "warning",
                "slide_id": sid,
                "message": "chart-like slide lacks legend labels",
            }
            issues.append(issue)
            slide_issues.append(issue)
            score -= 8

        slides_out.append({
            "slide_id": sid,
            "text_count": len(texts),
            "roles": dict(sorted(roles.items())),
            "has_chart": has_chart,
            "has_table": has_table,
            "has_legend": has_legend,
            "max_repeated_box_shape": max_repeated_shape,
            "quality_score": max(score, 0),
            "issue_count": len(slide_issues),
        })

    status = "pass"
    if any(issue["level"] == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "status": status,
        "layered_deck": str(layered_path),
        "slide_count": len(deck.get("slides") or []),
        "issue_count": len(issues),
        "issues": issues,
        "slides": slides_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layered_deck", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    report = audit(args.layered_deck)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail" or (args.fail_on_warn and report["status"] == "warn"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
