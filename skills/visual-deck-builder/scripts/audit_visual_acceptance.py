#!/usr/bin/env python3
"""Gate final deck delivery on visual review evidence.

This audit consumes a human or multimodal review JSON. It is intentionally not
an aesthetic classifier. The goal is to make visual inspection auditable, so a
deck cannot pass release only because structural scripts succeeded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = [
    "target_quality",
    "preview_readability",
    "text_rendering",
    "layer_fidelity",
    "icon_coverage",
    "semantic_drift",
    "overall",
]

VALID_STATUS = {"pass", "warn", "fail", "blocked"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2, "blocked": 3}.get(status, 2)


def _normalize_status(value: object) -> str:
    status = str(value or "").strip().lower()
    return status if status in VALID_STATUS else "fail"


def audit(
    review_path: Path,
    compare_path: Path | None = None,
    max_changed32: float = 0.12,
    max_changed64: float = 0.08,
    max_mean_abs: float = 20.0,
) -> dict:
    review = _load_json(review_path)
    issues: list[dict] = []
    slides_out: list[dict] = []

    slides = review.get("slides")
    if not isinstance(slides, list) or not slides:
        issues.append({"level": "error", "message": "review JSON must contain non-empty slides[]"})
        slides = []

    for index, slide in enumerate(slides, start=1):
        sid = str(slide.get("slide_id") or index)
        slide_issues: list[dict] = []
        statuses: dict[str, str] = {}

        for field in REQUIRED_FIELDS:
            if field not in slide:
                issue = {"level": "error", "slide_id": sid, "message": f"missing visual review field: {field}"}
                issues.append(issue)
                slide_issues.append(issue)
                statuses[field] = "fail"
                continue
            status = _normalize_status(slide.get(field))
            statuses[field] = status
            if status in {"fail", "blocked"}:
                issue = {
                    "level": "error",
                    "slide_id": sid,
                    "message": f"visual review {field} is {status}",
                }
                issues.append(issue)
                slide_issues.append(issue)
            elif status == "warn":
                issue = {
                    "level": "warning",
                    "slide_id": sid,
                    "message": f"visual review {field} is warn",
                }
                issues.append(issue)
                slide_issues.append(issue)

        notes = slide.get("notes")
        if not isinstance(notes, list) or not any(str(item).strip() for item in notes):
            issue = {"level": "warning", "slide_id": sid, "message": "visual review lacks reviewer notes"}
            issues.append(issue)
            slide_issues.append(issue)

        slides_out.append({
            "slide_id": sid,
            "statuses": statuses,
            "worst_status": max(statuses.values(), key=_status_rank) if statuses else "fail",
            "issue_count": len(slide_issues),
        })

    compare = None
    if compare_path:
        compare = _load_json(compare_path)
        changed32 = float(compare.get("changed_pixel_fraction_threshold_32", 0.0))
        changed64 = float(compare.get("changed_pixel_fraction_threshold_64", 0.0))
        mean_abs = float(compare.get("mean_abs_diff_0_255", 0.0))
        if changed32 > max_changed32:
            issues.append({
                "level": "warning",
                "message": "visual compare changed_pixel_fraction_threshold_32 exceeds threshold",
                "details": {"actual": changed32, "threshold": max_changed32},
            })
        if changed64 > max_changed64:
            issues.append({
                "level": "warning",
                "message": "visual compare changed_pixel_fraction_threshold_64 exceeds threshold",
                "details": {"actual": changed64, "threshold": max_changed64},
            })
        if mean_abs > max_mean_abs:
            issues.append({
                "level": "warning",
                "message": "visual compare mean_abs_diff_0_255 exceeds threshold",
                "details": {"actual": mean_abs, "threshold": max_mean_abs},
            })

    status = "pass"
    if any(issue["level"] == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "status": status,
        "review": str(review_path),
        "compare_report": str(compare_path) if compare_path else None,
        "issue_count": len(issues),
        "issues": issues,
        "slides": slides_out,
        "compare_metrics": {
            "changed_pixel_fraction_threshold_32": compare.get("changed_pixel_fraction_threshold_32") if compare else None,
            "changed_pixel_fraction_threshold_64": compare.get("changed_pixel_fraction_threshold_64") if compare else None,
            "mean_abs_diff_0_255": compare.get("mean_abs_diff_0_255") if compare else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_review", type=Path)
    parser.add_argument("--compare", type=Path, help="Optional visual_compare_qa.py report.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument("--max-changed32", type=float, default=0.12)
    parser.add_argument("--max-changed64", type=float, default=0.08)
    parser.add_argument("--max-mean-abs", type=float, default=20.0)
    args = parser.parse_args()

    report = audit(
        args.visual_review,
        compare_path=args.compare,
        max_changed32=args.max_changed32,
        max_changed64=args.max_changed64,
        max_mean_abs=args.max_mean_abs,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail" or (args.fail_on_warn and report["status"] == "warn"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
