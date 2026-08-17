import argparse
import json
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_manifest_paths(paths):
    collected = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            collected.extend(sorted(path.rglob("figure_manifest.json")))
        elif path.name == "figure_manifest.json":
            collected.append(path)
    seen = set()
    unique = []
    for path in collected:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path.resolve())
    return unique


def average_manual_axis_score(manifest):
    axes = ((manifest.get("quality") or {}).get("quality_review_axes") or [])
    scores = []
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        score = axis.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def semantic_review_summary(manifest):
    quality = manifest.get("quality") or {}
    summary = quality.get("semantic_review_summary")
    if isinstance(summary, dict):
        return summary
    review = quality.get("semantic_review") or manifest.get("semantic_review")
    if isinstance(review, dict):
        return {
            "status": review.get("status"),
            "provider": review.get("provider"),
            "model": review.get("model"),
            "overall_ready_to_ship": review.get("overall_ready_to_ship"),
            "recommended_next_action": review.get("recommended_next_action"),
            "missing_required_labels": review.get("missing_required_labels"),
            "forbidden_labels_seen": review.get("forbidden_labels_seen"),
        }
    return {}


def score_manifest(path, manifest):
    status = manifest.get("status")
    output = manifest.get("output") or {}
    quality = manifest.get("quality") or {}
    machine_summary = quality.get("machine_check_summary") or {}
    blocker = manifest.get("blocker")
    manual_avg = average_manual_axis_score(manifest)
    semantic_summary = semantic_review_summary(manifest)

    score = 0.0
    reasons = []
    if status == "completed":
        score += 50
        reasons.append("completed run")
    elif status == "blocked":
        score -= 50
        reasons.append("blocked run")
    if output.get("image_path"):
        score += 20
        reasons.append("has image")
    if machine_summary.get("status") == "passed":
        score += 15
        reasons.append("machine check passed")
    elif machine_summary.get("status") == "failed":
        score -= 20
        reasons.append("machine check failed")
    if manual_avg is not None:
        score += manual_avg * 3
        reasons.append(f"manual axis average {manual_avg}")
    if semantic_summary.get("status") == "completed":
        score += 10
        model = semantic_summary.get("model") or "semantic review"
        reasons.append(f"semantic review completed: {model}")
    if semantic_summary.get("overall_ready_to_ship") is True:
        score += 10
        reasons.append("semantic review ready to ship")
    elif semantic_summary.get("overall_ready_to_ship") is False:
        score -= 25
        reasons.append("semantic review not ready to ship")
    action = semantic_summary.get("recommended_next_action")
    if action == "deliver":
        score += 5
        reasons.append("semantic review recommends deliver")
    elif action in {"edit", "regenerate", "manual_review"}:
        score -= 10
        reasons.append(f"semantic review recommends {action}")
    missing_labels = semantic_summary.get("missing_required_labels") or []
    forbidden_seen = semantic_summary.get("forbidden_labels_seen") or []
    if missing_labels:
        score -= 8 * len(missing_labels)
        reasons.append(f"missing required labels: {len(missing_labels)}")
    if forbidden_seen:
        score -= 8 * len(forbidden_seen)
        reasons.append(f"forbidden labels seen: {len(forbidden_seen)}")
    if blocker:
        score -= 20
        reasons.append(f"blocker: {blocker.get('reason')}")

    return {
        "manifest_path": str(path),
        "run_id": manifest.get("run_id"),
        "figure_id": manifest.get("figure_id"),
        "status": status,
        "score": round(score, 2),
        "manual_axis_average": manual_avg,
        "semantic_review": semantic_summary or None,
        "image_path": output.get("image_path"),
        "reasons": reasons,
    }


def select_variant(manifest_paths):
    scored = [score_manifest(path, read_json(path)) for path in manifest_paths]
    scored.sort(key=lambda item: (item["score"], item.get("run_id") or ""), reverse=True)
    return {
        "schema": "giiisp_variant_selection_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best": scored[0] if scored else None,
        "candidates": scored,
    }


def main():
    parser = argparse.ArgumentParser(description="Select the best Giiisp figure variant from multiple manifests.")
    parser.add_argument("--input", nargs="+", required=True, help="Manifest files or directories containing figure_manifest.json files.")
    parser.add_argument("--out", help="Optional variant_selection.json output path.")
    args = parser.parse_args()

    manifest_paths = collect_manifest_paths(args.input)
    if not manifest_paths:
        raise SystemExit("No figure_manifest.json files found.")
    selection = select_variant(manifest_paths)
    if args.out:
        Path(args.out).write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
