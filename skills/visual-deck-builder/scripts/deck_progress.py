import argparse
import json
from datetime import datetime
from pathlib import Path


def now():
    return datetime.now().isoformat(timespec="seconds")


def clock():
    return datetime.now().strftime("%H:%M")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def count_items(value):
    return len(value) if isinstance(value, list) else 0


def load_slides(run_dir):
    spec = read_json(Path(run_dir) / "slide_spec.json") or {}
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        return spec.get("slides") or []
    return []


def build_status(run_dir):
    run_dir = Path(run_dir)
    slides = load_slides(run_dir)
    manifest = read_json(run_dir / "render_manifest.json") or {}
    deck_json = read_json(run_dir / "deck.json") or {}
    image_audit = read_json(run_dir / "qa" / "image-only-pptx.json") or {}
    visible_audit = read_json(run_dir / "qa" / "visible-text-review-audit.json") or {}
    workflow_events = run_dir / "stream_events.jsonl"

    manifest_slides = manifest.get("slides") if isinstance(manifest, dict) else []
    completed_images = [
        item
        for item in (manifest_slides or [])
        if isinstance(item, dict) and item.get("status") == "completed"
    ]
    blocked_images = [
        item
        for item in (manifest_slides or [])
        if isinstance(item, dict) and item.get("status") == "blocked"
    ]
    pptx_candidates = sorted((run_dir / "out").glob("*.pptx")) if (run_dir / "out").exists() else []
    previews = sorted((run_dir / "previews").glob("*")) if (run_dir / "previews").exists() else []

    status = "completed"
    blockers = []
    if blocked_images:
        status = "blocked"
        blockers.append("有页面没有拿到有效图片")
    elif slides and len(completed_images) < len(slides):
        status = "partial"
    elif not pptx_candidates:
        status = "partial"
    elif image_audit and image_audit.get("status") == "fail":
        status = "blocked"
        blockers.append("PPT 文件结构检查未通过")
    elif visible_audit and visible_audit.get("status") == "fail":
        status = "partial"
        blockers.append("有页面复查结果需要人工确认")
    elif visible_audit and visible_audit.get("status") == "warn":
        status = "partial"
        blockers.append("有页面复查结果需要人工确认")

    return {
        "schema": "visual_deck_workflow_status_v1",
        "created_at": now(),
        "run_dir": str(run_dir),
        "status": status,
        "slide_count": len(slides),
        "completed_image_count": len(completed_images),
        "blocked_image_count": len(blocked_images),
        "pptx": str(pptx_candidates[-1]) if pptx_candidates else None,
        "preview_count": len(previews),
        "artifacts": {
            "slide_spec": str(run_dir / "slide_spec.json") if (run_dir / "slide_spec.json").exists() else None,
            "render_manifest": str(run_dir / "render_manifest.json") if (run_dir / "render_manifest.json").exists() else None,
            "deck_json": str(run_dir / "deck.json") if (run_dir / "deck.json").exists() else None,
            "image_only_audit": str(run_dir / "qa" / "image-only-pptx.json")
            if (run_dir / "qa" / "image-only-pptx.json").exists()
            else None,
            "visible_text_audit": str(run_dir / "qa" / "visible-text-review-audit.json")
            if (run_dir / "qa" / "visible-text-review-audit.json").exists()
            else None,
            "stream_events": str(workflow_events) if workflow_events.exists() else None,
        },
        "blockers": blockers,
    }


def emit(args):
    run_dir = Path(args.run_dir)
    event = {
        "schema": "visual_deck_stream_event_v1",
        "created_at": now(),
        "event": args.event,
        "title": args.title,
        "message": args.message,
        "slide_id": args.slide_id,
        "slide_title": args.slide_title,
        "data": json.loads(args.data) if args.data else {},
    }
    append_jsonl(run_dir / "stream_events.jsonl", event)
    print(f"[{clock()} | {args.title}]")
    print(args.message)
    print()


def status(args):
    result = build_status(args.run_dir)
    out = Path(args.out) if args.out else Path(args.run_dir) / "deck_workflow_status.json"
    write_json(out, result)
    if args.print_summary:
        if result["status"] == "completed":
            message = f"整套 PPT 已完成：{result['slide_count']} 页图片、PPT 文件和审查记录已整理。"
        elif result["status"] == "partial":
            message = f"PPT 处于部分完成状态：{result['completed_image_count']}/{result['slide_count']} 页图片已完成。"
        else:
            message = "PPT 流程已停下；" + ("；".join(result["blockers"]) if result["blockers"] else "需要查看 blocker 或 QA 报告。")
        print(f"[{clock()} | 流程状态]")
        print(message)
        print()


def main():
    parser = argparse.ArgumentParser(description="Record user-visible progress and deck workflow status.")
    sub = parser.add_subparsers(dest="command", required=True)

    emit_parser = sub.add_parser("emit")
    emit_parser.add_argument("--run-dir", required=True)
    emit_parser.add_argument("--event", required=True)
    emit_parser.add_argument("--title", required=True)
    emit_parser.add_argument("--message", required=True)
    emit_parser.add_argument("--slide-id")
    emit_parser.add_argument("--slide-title")
    emit_parser.add_argument("--data")
    emit_parser.set_defaults(func=emit)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--run-dir", required=True)
    status_parser.add_argument("--out")
    status_parser.add_argument("--print-summary", action="store_true")
    status_parser.set_defaults(func=status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
