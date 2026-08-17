import argparse
import json
from datetime import datetime
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def read_text(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"_read_error: {exc}"


def first_existing(run_dir, names):
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def build_manifest(run_dir, source_run=None, feedback=None):
    run_dir = Path(run_dir).resolve()
    request = read_json(run_dir / "request.json")
    run_input = read_json(run_dir / "run_input.json")
    metadata = read_json(run_dir / "metadata.json")
    response = read_json(run_dir / "response.json")
    check = read_json(run_dir / "check.json")
    semantic_review = read_json(run_dir / "semantic_review.json")
    blocker = read_json(run_dir / "blocker.json")
    figure_spec = read_json(run_dir / "figure_spec.json")
    manual_review = read_text(run_dir / "manual_review.md")
    image = first_existing(run_dir, ["generated_image.png", "generated_image.jpg", "generated_image.jpeg", "generated_image.webp"])
    source_run_path = source_run or read_text(run_dir / "source_run.txt")
    run_input_dict = run_input if isinstance(run_input, dict) else {}
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    check_dict = check if isinstance(check, dict) else {}
    semantic_review_dict = semantic_review if isinstance(semantic_review, dict) else {}
    figure_spec_dict = figure_spec if isinstance(figure_spec, dict) else run_input_dict.get("figure_spec") or {}

    body = (request or {}).get("body", {}) if isinstance(request, dict) else {}
    if not body and isinstance(run_input, dict):
        body = run_input.get("request_body_summary") or {}
    image_sha256 = None
    if image:
        try:
            import hashlib

            digest = hashlib.sha256()
            with image.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            image_sha256 = digest.hexdigest()
        except Exception:
            image_sha256 = None
    inferred_status = metadata_dict.get("status")
    if not inferred_status:
        if blocker:
            inferred_status = "blocked"
        elif image:
            inferred_status = "completed"
    manifest = {
        "schema": "giiisp_figure_manifest_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "figure_id": run_dir.name,
        "run_dir": str(run_dir),
        "source_run": str(source_run_path).strip() if source_run_path else None,
        "feedback": feedback,
        "package_role": "single_figure",
        "task": {
            "kind": "scientific_figure",
            "figure_kind": run_input_dict.get("figure_kind"),
            "caption": run_input_dict.get("caption"),
            "label": run_input_dict.get("caption"),
            "panel": None,
            "intent": run_input_dict.get("communicative_intent"),
            "scientific_claim": run_input_dict.get("communicative_intent"),
            "source_context_excerpt": (run_input_dict.get("source_context") or "")[:500],
            "required_labels": run_input_dict.get("required_labels"),
            "forbidden_labels": run_input_dict.get("forbidden_labels"),
            "layout_brief": run_input_dict.get("layout_brief") or figure_spec_dict.get("layout_brief"),
            "style_brief": run_input_dict.get("style_brief"),
            "reference_role": run_input_dict.get("reference_role") or figure_spec_dict.get("reference_role"),
            "figure_spec": figure_spec_dict,
            "prompt": body.get("prompt"),
            "negative_prompt": body.get("negativePrompt"),
            "aspect_ratio": body.get("aspectRatio"),
            "image_size": body.get("imageSize"),
            "number_of_images": body.get("numberOfImages"),
            "response_modalities": body.get("responseModalities"),
            "output_mime_type": body.get("outputMimeType"),
        },
        "artifacts": {
            "request": str(run_dir / "request.json") if (run_dir / "request.json").exists() else None,
            "response": str(run_dir / "response.json") if (run_dir / "response.json").exists() else None,
            "poll_history": str(run_dir / "poll_history.json") if (run_dir / "poll_history.json").exists() else None,
            "image": str(image) if image else None,
            "check": str(run_dir / "check.json") if (run_dir / "check.json").exists() else None,
            "semantic_review": str(run_dir / "semantic_review.json") if (run_dir / "semantic_review.json").exists() else None,
            "manual_review": str(run_dir / "manual_review.md") if (run_dir / "manual_review.md").exists() else None,
            "blocker": str(run_dir / "blocker.json") if (run_dir / "blocker.json").exists() else None,
            "metadata": str(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else None,
            "run_input": str(run_dir / "run_input.json") if (run_dir / "run_input.json").exists() else None,
            "figure_spec": str(run_dir / "figure_spec.json") if (run_dir / "figure_spec.json").exists() else None,
        },
        "output": {
            "image_path": str(image) if image else None,
            "image_sha256": image_sha256,
            "mime_type": check_dict.get("mime_type"),
            "width": check_dict.get("width"),
            "height": check_dict.get("height"),
        },
        "status": inferred_status,
        "attempts": {
            "poll_count": metadata_dict.get("poll_count"),
            "iterations_completed": metadata_dict.get("iterations_completed"),
            "generation_requests": (metadata_dict.get("attempts") or {}).get("generation_requests"),
            "image_downloads": (metadata_dict.get("attempts") or {}).get("image_downloads"),
        },
        "lineage": {
            "source_run": str(source_run_path).strip() if source_run_path else run_input_dict.get("source_run"),
            "continue_from": run_input_dict.get("continue_from"),
            "parent_manifest": (run_input_dict.get("continue_from") or {}).get("parent_manifest_path")
            if isinstance(run_input_dict.get("continue_from"), dict)
            else None,
            "feedback": feedback or run_input_dict.get("feedback"),
            "continued_from_image": (run_input_dict.get("continue_from") or {}).get("source_image_path")
            if isinstance(run_input_dict.get("continue_from"), dict)
            else None,
        },
        "quality": {
            "machine_check": check,
            "machine_check_summary": check_dict.get("machine_check"),
            "quality_review_axes": check_dict.get("quality_review_axes"),
            "semantic_review": semantic_review,
            "semantic_review_summary": {
                "status": semantic_review_dict.get("status"),
                "provider": semantic_review_dict.get("provider"),
                "model": semantic_review_dict.get("model"),
                "overall_ready_to_ship": semantic_review_dict.get("overall_ready_to_ship"),
                "recommended_next_action": semantic_review_dict.get("recommended_next_action"),
                "missing_required_labels": semantic_review_dict.get("missing_required_labels"),
                "forbidden_labels_seen": semantic_review_dict.get("forbidden_labels_seen"),
            }
            if semantic_review
            else None,
            "manual_review_summary": manual_review[:500] if manual_review else None,
            "label_accuracy": None,
            "layout_quality": None,
            "needs_regeneration": True if blocker else None,
        },
        "deliverables": {
            "final_image": str(image) if image else None,
            "review_report": str(run_dir / "manual_review.md") if (run_dir / "manual_review.md").exists() else None,
            "request_response_evidence": [
                str(path)
                for path in [run_dir / "request.json", run_dir / "response.json", run_dir / "poll_history.json"]
                if path.exists()
            ],
        },
        "machine_check": check,
        "semantic_review": semantic_review,
        "blocker": blocker,
        "manual_review_excerpt": manual_review[:1000] if manual_review else None,
        "raw_response_summary": {
            "status_code": response.get("status_code") if isinstance(response, dict) else None,
            "has_json": isinstance(response, dict) and isinstance(response.get("json"), dict),
        },
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a PaperBanana-style manifest for one Giiisp figure run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--feedback")
    parser.add_argument("--out")
    args = parser.parse_args()

    manifest = build_manifest(args.run_dir, source_run=args.source_run, feedback=args.feedback)
    out = Path(args.out) if args.out else Path(args.run_dir) / "figure_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
