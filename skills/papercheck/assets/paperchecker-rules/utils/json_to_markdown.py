import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.report_markdown import build_markdown_report  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 PaperChecker JSON 报告转换为 Markdown 文件")
    parser.add_argument("--json", required=True, help="JSON 报告文件路径")
    parser.add_argument("--output-dir", default="reports_md", help="Markdown 输出目录")
    parser.add_argument("--output-name", help="自定义输出文件名（可选）")
    parser.add_argument(
        "--workspace",
        default=".",
        help="工作区根目录（用于解析相对路径）",
    )
    return parser.parse_args()


def convert(json_path: Path, output_dir: Path, output_name: Optional[str] = None) -> Path:
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as src:
        report = json.load(src)

    markdown = build_markdown_report(report)

    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{json_path.stem}_{timestamp}.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    with open(output_path, "w", encoding="utf-8") as target:
        target.write(markdown)
    return output_path


def main():
    args = parse_arguments()
    workspace = Path(args.workspace).resolve()
    json_path = (workspace / args.json).resolve() if not Path(args.json).is_absolute() else Path(args.json)
    output_dir = (workspace / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    try:
        output_path = convert(json_path, output_dir, args.output_name)
        print(json.dumps({"markdown_path": str(output_path)}, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - CLI 运行
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
