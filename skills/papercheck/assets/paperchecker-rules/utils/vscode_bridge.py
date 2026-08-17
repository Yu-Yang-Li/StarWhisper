import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.report_markdown import build_markdown_report  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VS Code 插件桥接脚本")
    parser.add_argument("--document", required=True, help="需要分析的文档路径")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="后端服务地址")
    parser.add_argument("--workspace", default=".", help="工作区根目录")
    parser.add_argument("--json-dir", default="reports", help="JSON 报告输出目录（相对 workspace）")
    parser.add_argument("--md-dir", default="reports_md", help="Markdown 输出目录（相对 workspace）")
    parser.add_argument("--timeout", type=int, default=300, help="请求超时时间（秒）")
    parser.add_argument("--wait", type=int, default=2, help="在上传前等待的秒数（便于服务启动）")
    return parser.parse_args()


def ensure_directory(base: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    resolved = path if path.is_absolute() else (base / path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def upload_document(document_path: Path, server_url: str, timeout: int) -> dict:
    endpoint = server_url.rstrip("/") + "/api/full-report"
    with open(document_path, "rb") as payload:
        files = {"file": (document_path.name, payload, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        response = requests.post(endpoint, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json()


def save_outputs(report: dict, json_dir: Path, md_dir: Path, base_name: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = json_dir / f"{base_name}_{timestamp}.json"
    md_path = md_dir / f"{base_name}_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)

    md_content = build_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(md_content)

    return json_path, md_path


def main():
    args = parse_arguments()
    workspace = Path(args.workspace).resolve()
    document = Path(args.document).resolve()

    if not document.exists():
        print(json.dumps({"error": f"找不到文档 {document}"}), file=sys.stderr)
        sys.exit(1)

    time.sleep(max(args.wait, 0))

    try:
        report = upload_document(document, args.server, args.timeout)
    except Exception as exc:  # pragma: no cover - CLI 通道
        print(json.dumps({"error": f"上传或分析失败: {exc}"}), file=sys.stderr)
        sys.exit(2)

    json_dir = ensure_directory(workspace, args.json_dir)
    md_dir = ensure_directory(workspace, args.md_dir)

    json_path, md_path = save_outputs(report, json_dir, md_dir, document.stem)

    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "match_rate": report.get("match_rate"),
                "document": report.get("document", document.name),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

