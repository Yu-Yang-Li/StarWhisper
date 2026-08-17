#!/usr/bin/env python3
"""Validate visual-deck-builder specs, layered decks, and layer manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


PLACEHOLDER_TERMS = {
    "lorem ipsum",
    "placeholder",
    "todo",
    "待补充",
    "占位",
}
REQUIRED_DECK_FIELDS = ["title", "language", "aspect_ratio", "route", "style_brief"]
REQUIRED_SLIDE_FIELDS = [
    "slide_id",
    "purpose",
    "title",
    "body_text",
    "must_show",
    "visual_brief",
    "editable_strategy",
    "layer_prompts",
    "editable_text",
]
VALID_ASPECT_RATIOS = {"16:9", "4:3", "3:2", "1:1"}
VALID_STATUSES = {"planned", "visual_target_rendered", "extracted", "layers_rendered", "composed", "qa_passed", "needs_retry", "blocked"}
INVALID_BACKENDS = {"programmatic", "pil", "svg", "html", "canvas", "matplotlib", "screenshot"}
VALID_EDITABLE_STRATEGIES = {"layered_editable"}
REQUIRED_LAYER_PROMPTS = ["visual_target", "background", "frame", "icons"]
REQUIRED_LAYER_ASSETS = ["background", "frame"]
REQUIRED_MANIFEST_LAYERS = ["visual_target", "background", "frame"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = base / p
    return p


def _is_missing(value) -> bool:
    return value is None or value == "" or value == []


def _validate_box(box: dict, slide_id: str, item_index: int, issues: list[dict]) -> None:
    if not isinstance(box, dict):
        issues.append({
            "level": "error",
            "slide_id": slide_id,
            "message": f"editable_text[{item_index}] box must be an object",
        })
        return
    for key in ["x", "y", "w", "h"]:
        if key not in box:
            issues.append({
                "level": "error",
                "slide_id": slide_id,
                "message": f"editable_text[{item_index}] box missing {key}",
            })
            continue
        try:
            value = float(box[key])
        except (TypeError, ValueError):
            issues.append({
                "level": "error",
                "slide_id": slide_id,
                "message": f"editable_text[{item_index}] box {key} is not numeric",
            })
            continue
        if not 0 <= value <= 1:
            issues.append({
                "level": "error",
                "slide_id": slide_id,
                "message": f"editable_text[{item_index}] box {key} outside [0, 1]",
            })
    try:
        x = float(box.get("x"))
        y = float(box.get("y"))
        w = float(box.get("w"))
        h = float(box.get("h"))
    except (TypeError, ValueError):
        return
    if w <= 0 or h <= 0:
        issues.append({
            "level": "error",
            "slide_id": slide_id,
            "message": f"editable_text[{item_index}] box width/height must be positive",
        })
    if x + w > 1 or y + h > 1:
        issues.append({
            "level": "error",
            "slide_id": slide_id,
            "message": f"editable_text[{item_index}] box extends beyond slide canvas",
        })


def _validate_positioned_box(item: dict, label: str, issues: list[dict], slide_id: str | None = None) -> None:
    for key in ["x", "y", "w", "h"]:
        if key not in item:
            issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} missing {key}"})
            continue
        try:
            value = float(item[key])
        except (TypeError, ValueError):
            issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} {key} is not numeric"})
            continue
        if not 0 <= value <= 1:
            issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} {key} outside [0, 1]"})
    try:
        x = float(item.get("x"))
        y = float(item.get("y"))
        w = float(item.get("w"))
        h = float(item.get("h"))
    except (TypeError, ValueError):
        return
    if w <= 0 or h <= 0:
        issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} width/height must be positive"})
    if x + w > 1 or y + h > 1:
        issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} extends beyond slide canvas"})


def _validate_image(path: Path, label: str, issues: list[dict], slide_id: str | None = None) -> None:
    if not path.exists():
        issues.append({"level": "error", "slide_id": slide_id, "message": f"{label} not found: {path}"})
        return
    try:
        with Image.open(path) as im:
            w, h = im.size
        if w < 256 or h < 256:
            issues.append({"level": "warning", "slide_id": slide_id, "message": f"{label} is small: {w}x{h}"})
    except Exception as exc:
        issues.append({"level": "error", "slide_id": slide_id, "message": f"cannot read {label}: {exc}"})


def _manifest_entries(manifest: dict | list) -> list[dict]:
    if isinstance(manifest, list):
        return [entry for entry in manifest if isinstance(entry, dict)]
    entries = manifest.get("slides") if isinstance(manifest, dict) else []
    return [entry for entry in (entries or []) if isinstance(entry, dict)]


def _validate_layer_manifest(manifest_path: Path | None, base: Path, expected_ids: set[str], issues: list[dict]) -> None:
    if not manifest_path:
        issues.append({"level": "error", "message": "missing layer_manifest.json"})
        return
    if not manifest_path.exists():
        issues.append({"level": "error", "message": f"layer manifest not found: {manifest_path}"})
        return

    manifest = _load_json(manifest_path)
    entries = _manifest_entries(manifest)
    manifest_ids = {str(entry.get("slide_id")) for entry in entries}
    for sid in expected_ids:
        if sid not in manifest_ids:
            issues.append({"level": "error", "slide_id": sid, "message": "slide missing from layer manifest"})

    for entry in entries:
        sid = str(entry.get("slide_id") or "")
        layers = entry.get("layers") or []
        if not isinstance(layers, list):
            issues.append({"level": "error", "slide_id": sid, "message": "layer manifest layers must be a list"})
            continue
        seen_layers = {str(layer.get("layer")) for layer in layers if isinstance(layer, dict)}
        for required in REQUIRED_MANIFEST_LAYERS:
            if required not in seen_layers:
                issues.append({"level": "error", "slide_id": sid, "message": f"layer manifest missing {required}"})
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            backend = str(layer.get("backend") or "").strip().lower()
            if not backend or backend in INVALID_BACKENDS:
                issues.append({"level": "error", "slide_id": sid, "message": f"invalid backend for layer {layer.get('layer')}"})
            if str(layer.get("status") or "").strip().lower() == "blocked":
                issues.append({"level": "error", "slide_id": sid, "message": f"layer {layer.get('layer')} is blocked"})
            for field in ["prompt_file", "generated_source", "copied_to"]:
                if _is_missing(layer.get(field)):
                    issues.append({"level": "error", "slide_id": sid, "message": f"layer {layer.get('layer')} missing {field}"})
            copied = _resolve(base, layer.get("copied_to"))
            if copied:
                _validate_image(copied, f"layer {layer.get('layer')}", issues, sid)


def _validate_layered_deck(layered_path: Path | None, spec_base: Path, expected_ids: set[str], issues: list[dict]) -> None:
    if not layered_path:
        issues.append({"level": "error", "message": "missing layered_deck.json"})
        return
    if not layered_path.exists():
        issues.append({"level": "error", "message": f"layered deck not found: {layered_path}"})
        return

    deck = _load_json(layered_path)
    base = Path(deck.get("assets_dir") or layered_path.parent)
    if not base.is_absolute():
        base = layered_path.parent / base
    slides = deck.get("slides") or []
    if not slides:
        issues.append({"level": "error", "message": "layered_deck.json has no slides"})
        return
    layered_ids = set()
    for index, slide in enumerate(slides, start=1):
        sid = str(slide.get("slide_id") or f"index-{index}")
        layered_ids.add(sid)
        for required in REQUIRED_LAYER_ASSETS:
            path = _resolve(base, slide.get(required))
            if not path:
                issues.append({"level": "error", "slide_id": sid, "message": f"layered slide missing {required}"})
            else:
                _validate_image(path, required, issues, sid)
        icons = slide.get("icons") or []
        if not isinstance(icons, list):
            issues.append({"level": "error", "slide_id": sid, "message": "icons must be a list"})
            icons = []
        for item_index, icon in enumerate(icons):
            if not isinstance(icon, dict):
                issues.append({"level": "error", "slide_id": sid, "message": f"icons[{item_index}] must be an object"})
                continue
            path = _resolve(base, icon.get("file"))
            if not path:
                issues.append({"level": "error", "slide_id": sid, "message": f"icons[{item_index}] missing file"})
            else:
                _validate_image(path, f"icons[{item_index}]", issues, sid)
            _validate_positioned_box(icon, f"icons[{item_index}]", issues, sid)
        texts = slide.get("texts") or []
        if not isinstance(texts, list) or not texts:
            issues.append({"level": "error", "slide_id": sid, "message": "layered slide must contain real PPT texts"})
            texts = []
        for item_index, text in enumerate(texts):
            if not isinstance(text, dict):
                issues.append({"level": "error", "slide_id": sid, "message": f"texts[{item_index}] must be an object"})
                continue
            if not str(text.get("text") or "").strip():
                issues.append({"level": "error", "slide_id": sid, "message": f"texts[{item_index}] missing text"})
            _validate_positioned_box(text, f"texts[{item_index}]", issues, sid)
    for sid in expected_ids:
        if sid not in layered_ids:
            issues.append({"level": "error", "slide_id": sid, "message": "slide missing from layered_deck.json"})


def validate(
    spec_path: Path,
    layer_manifest_path: Path | None,
    layered_deck_path: Path | None = None,
    allow_planning: bool = False,
) -> dict:
    spec = _load_json(spec_path)
    base = spec_path.parent
    deck = spec.get("deck") or {}
    slides = spec.get("slides") or []
    issues: list[dict] = []

    for field in REQUIRED_DECK_FIELDS:
        if _is_missing(deck.get(field)):
            issues.append({"level": "error", "message": f"deck missing {field}"})
    if deck.get("aspect_ratio") and deck.get("aspect_ratio") not in VALID_ASPECT_RATIOS:
        issues.append({"level": "warning", "message": f"unsupported aspect_ratio: {deck.get('aspect_ratio')}"})

    routes = deck.get("route") or []
    if isinstance(routes, str):
        routes = [routes]
    if not isinstance(routes, list):
        issues.append({"level": "error", "message": "deck route must be a list or string"})
        routes = []
    route_set = {str(route) for route in routes}
    source_grounded = "source_grounded" in route_set
    style_reference = "style_reference" in route_set
    existing_slide_edit = "existing_slide_edit" in route_set
    deck_reference_guard = deck.get("reference_guard")
    deck_source_targets = deck.get("source_visual_targets")

    if not slides:
        issues.append({"level": "error", "message": "no slides in slide_spec.json"})

    seen_ids = set()
    for index, slide in enumerate(slides, start=1):
        sid = str(slide.get("slide_id") or f"index-{index}")
        if sid in seen_ids:
            issues.append({"level": "error", "slide_id": sid, "message": "duplicate slide_id"})
        seen_ids.add(sid)

        for field in REQUIRED_SLIDE_FIELDS:
            if field not in slide or (field != "editable_text" and _is_missing(slide.get(field))):
                issues.append({"level": "error", "slide_id": sid, "message": f"missing {field}"})
        if "status" in slide and slide.get("status") not in VALID_STATUSES:
            issues.append({"level": "error", "slide_id": sid, "message": f"invalid status: {slide.get('status')}"})

        strategy = str(slide.get("editable_strategy") or deck.get("editable_strategy") or "")
        if strategy not in VALID_EDITABLE_STRATEGIES:
            issues.append({"level": "error", "slide_id": sid, "message": f"invalid editable_strategy: {strategy}"})

        layer_prompts = slide.get("layer_prompts") or {}
        if not isinstance(layer_prompts, dict):
            issues.append({"level": "error", "slide_id": sid, "message": "layer_prompts must be an object"})
            layer_prompts = {}
        for layer in REQUIRED_LAYER_PROMPTS:
            if _is_missing(layer_prompts.get(layer)):
                issues.append({"level": "error", "slide_id": sid, "message": f"layer_prompts missing {layer}"})

        editable_text = slide.get("editable_text") or []
        if not isinstance(editable_text, list):
            issues.append({"level": "error", "slide_id": sid, "message": "editable_text must be a list"})
            editable_text = []
        for item_index, item in enumerate(editable_text):
            if not isinstance(item, dict):
                issues.append({
                    "level": "error",
                    "slide_id": sid,
                    "message": f"editable_text[{item_index}] must be an object",
                })
                continue
            _validate_box(item.get("box") or {}, sid, item_index, issues)

        if source_grounded and not slide.get("evidence"):
            issues.append({"level": "error", "slide_id": sid, "message": "source_grounded slide missing evidence"})

        if existing_slide_edit and not slide.get("source_visual_target") and not deck_source_targets:
            issues.append({"level": "error", "slide_id": sid, "message": "existing_slide_edit slide missing source_visual_target"})

        reference_guard = slide.get("reference_guard") or deck_reference_guard
        if style_reference and not reference_guard:
            issues.append({"level": "error", "slide_id": sid, "message": "style_reference slide missing reference_guard"})
        if isinstance(reference_guard, dict):
            forbidden_terms = reference_guard.get("forbidden_terms") or []
            content_text = " ".join([
                str(slide.get("title") or ""),
                " ".join(map(str, slide.get("body_text") or [])),
                " ".join(map(str, slide.get("must_show") or [])),
                " ".join(str(item.get("text") or "") for item in (editable_text or []) if isinstance(item, dict)),
            ]).lower()
            for term in forbidden_terms:
                term_text = str(term).strip().lower()
                if term_text and term_text in content_text:
                    issues.append({"level": "error", "slide_id": sid, "message": f"reference_guard forbidden term in content: {term}"})

        prompt_text = " ".join([
            str(slide.get("title") or ""),
            " ".join(map(str, slide.get("body_text") or [])),
            " ".join(str(v) for v in layer_prompts.values()),
        ]).lower()
        for term in PLACEHOLDER_TERMS:
            if term in prompt_text:
                issues.append({"level": "warning", "slide_id": sid, "message": f"placeholder-like term: {term}"})

    if not allow_planning:
        _validate_layered_deck(layered_deck_path, base, seen_ids, issues)
        _validate_layer_manifest(layer_manifest_path, base, seen_ids, issues)

    status = "pass"
    if any(i["level"] == "error" for i in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "status": status,
        "slide_count": len(slides),
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--manifest", type=Path, help="Layer manifest path")
    parser.add_argument("--layer-manifest", type=Path, help="Layer manifest path")
    parser.add_argument("--layers", type=Path, help="Path to layered_deck.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-planning", action="store_true", help="Permit planning-stage specs without layer assets")
    args = parser.parse_args()

    report = validate(args.spec, args.layer_manifest or args.manifest, args.layers, args.allow_planning)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
