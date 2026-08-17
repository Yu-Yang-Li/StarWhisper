import argparse
import json
from datetime import datetime
from pathlib import Path


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def build_package(manifest_paths, package_dir, title):
    package_dir = Path(package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    manifests = [read_json(path) for path in manifest_paths]
    package_id = package_dir.name

    plan_items = []
    report_items = []
    for index, (path, manifest) in enumerate(zip(manifest_paths, manifests), start=1):
        task = manifest.get("task") or {}
        output = manifest.get("output") or {}
        quality = manifest.get("quality") or {}
        semantic_summary = quality.get("semantic_review_summary") or {}
        item_id = manifest.get("figure_id") or f"figure_{index:02d}"
        plan_items.append(
            {
                "id": item_id,
                "index": index,
                "caption": task.get("caption"),
                "figure_kind": task.get("figure_kind"),
                "intent": task.get("intent"),
                "required_labels": task.get("required_labels"),
                "layout_brief": task.get("layout_brief"),
                "reference_role": task.get("reference_role"),
                "manifest_path": str(path),
                "figure_spec_path": (manifest.get("artifacts") or {}).get("figure_spec"),
                "semantic_review_path": (manifest.get("artifacts") or {}).get("semantic_review"),
            }
        )
        report_items.append(
            {
                "id": item_id,
                "index": index,
                "status": manifest.get("status"),
                "run_id": manifest.get("run_id"),
                "run_dir": manifest.get("run_dir"),
                "image_path": output.get("image_path"),
                "image_sha256": output.get("image_sha256"),
                "caption": task.get("caption"),
                "layout_brief": task.get("layout_brief"),
                "reference_role": task.get("reference_role"),
                "semantic_review_status": semantic_summary.get("status"),
                "semantic_review_model": semantic_summary.get("model"),
                "overall_ready_to_ship": semantic_summary.get("overall_ready_to_ship"),
                "recommended_next_action": semantic_summary.get("recommended_next_action"),
                "error": (manifest.get("blocker") or {}).get("reason") if manifest.get("blocker") else None,
                "manifest_path": str(path),
                "figure_spec_path": (manifest.get("artifacts") or {}).get("figure_spec"),
                "semantic_review_path": (manifest.get("artifacts") or {}).get("semantic_review"),
            }
        )

    plan = {
        "schema": "giiisp_figure_package_plan_v1",
        "created_at": now,
        "package_id": package_id,
        "title": title,
        "items": plan_items,
    }
    checkpoint = {
        "schema": "giiisp_figure_package_checkpoint_v1",
        "created_at": now,
        "updated_at": now,
        "package_id": package_id,
        "status": "completed" if all(item["status"] == "completed" for item in report_items) else "needs_review",
        "items": [
            {
                "id": item["id"],
                "status": item["status"],
                "run_id": item["run_id"],
                "manifest_path": item["manifest_path"],
                "error": item["error"],
            }
            for item in report_items
        ],
    }
    report = {
        "schema": "giiisp_figure_package_report_v1",
        "created_at": now,
        "package_id": package_id,
        "title": title,
        "status": checkpoint["status"],
        "total_figures": len(report_items),
        "completed_figures": sum(1 for item in report_items if item["status"] == "completed"),
        "blocked_figures": sum(1 for item in report_items if item["status"] == "blocked"),
        "items": report_items,
    }

    write_json(package_dir / "package_plan.json", plan)
    write_json(package_dir / "package_checkpoint.json", checkpoint)
    write_json(package_dir / "figure_package.json", report)
    return package_dir / "figure_package.json"


def main():
    parser = argparse.ArgumentParser(description="Build a PaperBanana-style package report from Giiisp figure manifests.")
    parser.add_argument("--input", nargs="+", required=True, help="Figure manifest files or directories containing figure_manifest.json files.")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--title", default="科研图像生成结果包")
    args = parser.parse_args()

    manifest_paths = collect_manifest_paths(args.input)
    if not manifest_paths:
        raise SystemExit("No figure_manifest.json files found.")
    report_path = build_package(manifest_paths, args.package_dir, args.title)
    print(str(report_path))


if __name__ == "__main__":
    main()
