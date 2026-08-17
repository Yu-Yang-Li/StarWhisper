import argparse
import base64
import hashlib
import json
from pathlib import Path

ENDPOINT = "http://images.sitianai.com/api/generate-async"


def infer_mime(path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    figure_spec = data.get("figure_spec") or {}
    return {
        "prompt": data.get("prompt", ""),
        "negative_prompt": data.get("negative_prompt", data.get("negativePrompt", "水印，模糊文字，错乱标签，低清晰度，广告风格")),
        "aspect_ratio": data.get("aspect_ratio", data.get("aspectRatio", "1:1")),
        "image_size": data.get("image_size", data.get("imageSize", "1K")),
        "number_of_images": data.get("number_of_images", data.get("numberOfImages", 1)),
        "reference_image": data.get("reference_image"),
        "source_run": data.get("source_run"),
        "source_context": data.get("source_context"),
        "caption": data.get("caption"),
        "intent": data.get("intent", data.get("communicative_intent")),
        "figure_kind": data.get("figure_kind", figure_spec.get("figure_kind", "workflow")),
        "required_labels": data.get("required_labels", figure_spec.get("required_labels", [])),
        "forbidden_labels": data.get("forbidden_labels", figure_spec.get("forbidden_labels", [])),
        "style_brief": data.get("style_brief", figure_spec.get("style_brief")),
        "feedback": data.get("feedback"),
        "reference_role": data.get("reference_role", figure_spec.get("reference_role", "")),
        "preserve_constraints": data.get("preserve_constraints", figure_spec.get("preserve_constraints", [])),
        "allowed_changes": data.get("allowed_changes", figure_spec.get("allowed_changes", [])),
        "disallowed_changes": data.get("disallowed_changes", figure_spec.get("disallowed_changes", [])),
        "layout_brief": data.get("layout_brief", figure_spec.get("layout_brief")),
    }


def merge_input_json(args):
    if not args.input_json:
        return
    auto = load_input_json(args.input_json)
    for name, value in auto.items():
        current = getattr(args, name)
        if current in (None, "", []):
            setattr(args, name, value)


def main():
    parser = argparse.ArgumentParser(description="Build a dry-run Giiisp Imagine generation request.")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default="水印，模糊文字，错乱标签，低清晰度，广告风格")
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--image-size", default="1K")
    parser.add_argument("--number-of-images", type=int, default=1)
    parser.add_argument("--reference-image", help="Optional local image for an edit/reference dry-run.")
    parser.add_argument("--source-run", help="Optional prior run directory for edit dry-runs.")
    parser.add_argument("--source-context", help="Original paper paragraph, method description, or user material behind the figure.")
    parser.add_argument("--caption", help="Short caption or figure title.")
    parser.add_argument("--intent", help="Scientific message this figure should communicate.")
    parser.add_argument("--figure-kind", default="workflow", help="workflow, mechanism schematic, method diagram, comparison figure, etc.")
    parser.add_argument("--required-labels", nargs="*", default=[], help="Labels that must appear in the figure.")
    parser.add_argument("--forbidden-labels", nargs="*", default=[], help="Labels or text patterns that must not appear.")
    parser.add_argument("--style-brief", help="Compact visual style brief.")
    parser.add_argument("--feedback", help="User feedback for edit/continue-style dry-runs.")
    parser.add_argument("--reference-role", default="", choices=["", "preserve_structure", "use_elements", "refine_sketch", "edit_image"], help="How to interpret the reference image, if present.")
    parser.add_argument("--preserve-constraints", nargs="*", default=[], help="Elements that should be preserved from the parent image/run.")
    parser.add_argument("--allowed-changes", nargs="*", default=[], help="Specific changes allowed in an edit run.")
    parser.add_argument("--disallowed-changes", nargs="*", default=[], help="Changes that must not happen in an edit run.")
    parser.add_argument("--layout-brief", help="Structured layout constraint such as horizontal four-step flow or 2x2 grid.")
    parser.add_argument("--input-json", help="Path to JSON containing prompt fields and optional figure_spec.")
    args = parser.parse_args()
    merge_input_json(args)

    if not args.prompt:
        parser.error("--prompt is required unless --input-json provides it")

    body = {
        "prompt": args.prompt.strip(),
        "negativePrompt": args.negative_prompt.strip(),
        "aspectRatio": args.aspect_ratio,
        "imageSize": args.image_size,
        "numberOfImages": max(1, min(4, args.number_of_images)),
        "responseModalities": ["IMAGE", "TEXT"],
        "outputMimeType": "image/png",
    }
    edit_note = None
    edit_metadata = {}
    if args.source_run:
        source_run = Path(args.source_run)
        edit_metadata["source_run"] = str(source_run)
        edit_metadata["source_run_exists"] = source_run.exists()
    if args.reference_image:
        reference_path = Path(args.reference_image)
        edit_metadata["reference_image_path"] = str(reference_path)
        edit_metadata["reference_image_exists"] = reference_path.exists()
        if reference_path.exists():
            body["imageBase64"] = base64.b64encode(reference_path.read_bytes()).decode("ascii")
            body["imageMimeType"] = infer_mime(reference_path)
            edit_metadata["reference_image_mime_type"] = body["imageMimeType"]
            edit_metadata["reference_image_size_bytes"] = reference_path.stat().st_size
            edit_metadata["reference_image_sha256"] = file_digest(reference_path)
            edit_note = "reference image encoded for edit/reference request"
        else:
            edit_note = "reference image path does not exist; request body left without imageBase64"

    display_body = dict(body)
    if "imageBase64" in display_body:
        display_body["imageBase64"] = "<redacted reference image base64>"

    output = {
        "method": "POST",
        "url": ENDPOINT,
        "headers": {
            "Content-Type": "application/json",
            "Referer": "http://images.sitianai.com/",
            "Authorization": "Bearer <giiisp_auth_token when available>",
        },
        "body": display_body,
        "safety": "dry-run only; no image generation request was sent",
        "run_input_preview": {
            "source_context": args.source_context,
            "caption": args.caption,
            "communicative_intent": args.intent or args.caption,
            "figure_kind": args.figure_kind,
            "required_labels": args.required_labels,
            "forbidden_labels": args.forbidden_labels,
            "style_brief": args.style_brief,
            "feedback": args.feedback,
        },
        "figure_spec_preview": {
            "schema": "giiisp_figure_spec_v1",
            "figure_kind": args.figure_kind,
            "caption": args.caption,
            "communicative_intent": args.intent or args.caption,
            "required_labels": args.required_labels,
            "forbidden_labels": args.forbidden_labels,
            "layout_brief": args.layout_brief,
            "style_brief": args.style_brief,
            "reference_role": args.reference_role,
            "preserve_constraints": args.preserve_constraints,
            "allowed_changes": args.allowed_changes,
            "disallowed_changes": args.disallowed_changes,
        },
    }
    if edit_note:
        output["edit_entry"] = edit_note
    if edit_metadata:
        output["edit_metadata"] = edit_metadata
        output["run_directory_rule"] = (
            "save edits in a new scientific_image_skill_runs/<session_slug>/edit_YYYYMMDD_HHMMSS "
            "directory and write source_run.txt; never overwrite the source run"
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
