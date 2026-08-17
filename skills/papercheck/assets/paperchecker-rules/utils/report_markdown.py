import json
import os
from datetime import datetime
from typing import Any, Dict, List
import re


def _md_value(value: Any) -> str:
    """Convert any value to a markdown-safe string."""
    if value is None:
        return "无"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _section_heading(title: str, level: int = 2) -> str:
    # 不再使用 Markdown 的 # 标题语法，改为加粗文本
    return f"**{title}**"


def _is_rule_code(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]+", text))


def _format_table(rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["无数据"]
    header = rows[0]
    lines = [
        "| " + " | ".join(_md_value(col) for col in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(_md_value(col) for col in row) + " |")
    return lines


def _format_list(items: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    if not items:
        return ["- 无"]
    lines: List[str] = []
    for item in items:
        parts = []
        for field in fields:
            value = item.get(field)
            if value:
                # 将字段名转换为中文
                if field == "original":
                    field_cn = "原文"
                elif field == "corrected":
                    field_cn = "修正"
                elif field == "reference":
                    field_cn = "规则编码" if _is_rule_code(value) else "参考文献"
                elif field == "correction_reason":
                    field_cn = "修正原因"
                elif field == "formatted":
                    field_cn = "格式化"
                elif field == "formatting_reason":
                    field_cn = "格式化原因"
                elif field == "text":
                    field_cn = "文本"
                elif field == "doi":
                    field_cn = "DOI"
                elif field == "url":
                    field_cn = "URL"
                elif field == "text":
                    field_cn = "文本"
                elif field == "issue_code":
                    field_cn = "规则编码"
                elif field == "severity":
                    field_cn = "严重级别"
                elif field == "reference_text":
                    field_cn = "参考文献条目"
                elif field == "message":
                    field_cn = "问题描述"
                elif field == "suggestion":
                    field_cn = "修正建议"
                elif field == "missing_citations":
                    field_cn = "缺失引文"
                elif field == "rule_strength":
                    field_cn = "规则强度"
                elif field == "confidence_tier":
                    field_cn = "置信度层级"
                elif field == "confidence_score":
                    field_cn = "置信度分数"
                else:
                    field_cn = field

                # 去掉[AUTH:]前缀
                if isinstance(value, str) and value.startswith("[AUTH:"):
                    value = value[6:-1]  # 去掉[AUTH:]前缀和后缀

                parts.append(f"**{field_cn}**: {_md_value(value)}")
        if parts:
            # 每个部分单独成行
            for part in parts:
                lines.append("- " + part)
    return lines or ["- 无"]


def _format_rule_grouped_items(items: List[Dict[str, Any]]) -> List[str]:
    """
    将规则驱动的格式化建议按原文条目分组，避免重复条目难以阅读。
    """
    if not items:
        return ["- 无"]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        original = str(item.get("original", "")).strip() or "（未提供原文）"
        grouped.setdefault(original, []).append(item)

    lines: List[str] = []
    for original, problems in grouped.items():
        lines.append(f"- **条目**: {_md_value(original)}")
        seen = set()
        for p in problems:
            reason = str(p.get("formatting_reason", "")).strip()
            suggestion = str(p.get("formatted", "")).strip()
            code = str(p.get("reference", "")).strip()
            key = (reason, suggestion, code)
            if key in seen:
                continue
            seen.add(key)
            if reason:
                lines.append(f"  - 问题: {_md_value(reason)}")
            if suggestion:
                lines.append(f"  - 建议: {_md_value(suggestion)}")
            if code:
                lines.append(f"  - 规则编码: {_md_value(code)}")
    return lines


def build_markdown_report(report: Dict[str, Any]) -> str:
    """
    将 PaperChecker JSON 报告转换为 Markdown 字符串。

    Args:
        report: PaperChecker 分析得到的 JSON 字典。

    Returns:
        Markdown 文本。
    """
    timestamp = report.get("test_date") or datetime.now().isoformat()
    document = report.get("document") or "未提供"

    # 提取文件名而不是完整路径
    document_filename = os.path.basename(document) if document != "未提供" else "未提供"

    # 格式化时间为中文格式
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if timestamp.endswith('Z') else datetime.fromisoformat(timestamp)
        formatted_time = dt.strftime("%Y年%m月%d日%H:%M:%S")
    except:
        # 如果时间格式有问题，使用原始时间戳
        formatted_time = timestamp

    lines: List[str] = [
        "PaperChecker 引用分析报告",
        "",
        f"- **文档**: {_md_value(document_filename)}",
        f"- **生成时间**: {_md_value(formatted_time)}",
    ]

    summary_rows = [
        ["指标", "数值"],
        ["引用总数", report.get("total_citations", "未知")],
        ["参考文献数", report.get("total_references", "未知")],
        ["匹配条目", report.get("matched_count", "未知")],
        ["未匹配条目", report.get("unmatched_count", "未知")],
        ["需要修正", report.get("corrected_count", "未知")],
        ["需要格式化", report.get("formatted_count", "未知")],
        ["正文引用格式化", report.get("citation_formatted_count", "未知")],
        ["参考文献格式化", report.get("reference_formatted_count", "未知")],
        ["参考文献著录问题", report.get("reference_format_issue_count", len(report.get("reference_format_issues", [])))],
        ["引文体系问题", report.get("citation_style_issue_count", len(report.get("citation_style_issues", [])))],
        ["强规则问题", report.get("strong_rule_issue_count", "未知")],
        ["启发式问题", report.get("heuristic_rule_issue_count", "未知")],
        ["高置信问题", report.get("high_confidence_issue_count", "未知")],
        ["中置信问题", report.get("medium_confidence_issue_count", "未知")],
        ["低置信问题", report.get("low_confidence_issue_count", "未知")],
        ["匹配率", report.get("match_rate", "未知")],
    ]

    lines += [
        "",
        _section_heading("整体统计"),
        "",
        *_format_table(summary_rows),
    ]

    lines += [
        "",
        _section_heading("需要修正的引用"),
        "",
        *_format_list(
            report.get("corrections_needed", []),
            ["original", "corrected", "reference", "correction_reason"],
        ),
        "",
        _section_heading("需要格式化的引用"),
    ]

    citation_like_formatting_items = report.get("citation_formatting_needed")
    if citation_like_formatting_items is None:
        citation_like_formatting_items = report.get("formatting_needed", [])
    reference_like_formatting_items = report.get("reference_formatting_needed", [])

    lines += [
        "",
        *_format_list(
            citation_like_formatting_items,
            ["original", "formatted", "reference", "formatting_reason"],
        ),
        "",
        _section_heading("需要格式化的参考文献（按条目归并）"),
        "",
        *_format_rule_grouped_items(reference_like_formatting_items),
        "",
        _section_heading("参考文献著录问题"),
        "",
        *_format_list(
            report.get("reference_format_issues", []),
            ["issue_code", "severity", "rule_strength", "confidence_tier", "confidence_score", "reference_text", "message", "suggestion"],
        ),
        "",
        _section_heading("引文体系问题"),
        "",
        *_format_list(
            report.get("citation_style_issues", []),
            ["issue_code", "severity", "rule_strength", "confidence_tier", "confidence_score", "missing_citations", "message", "suggestion"],
        ),
        "",
        _section_heading("未被引用的参考文献"),
        "",
        *_format_list(
            report.get("unused_references", []),
            ["text", "doi", "url"],
        ),
    ]

    results = report.get("results", [])
    if results:
        lines += ["", _section_heading("引用详情"), ""]
        for entry in results:
            idx = entry.get("citation_index", "?")
            lines.append(f"**引用 {idx}**")
            bullet_lines = [
                f"- 原始引用: {_md_value(entry.get('original_citation'))}",
                f"- 匹配状态: {_md_value('已匹配' if entry.get('matched') else '未匹配')}",
            ]
            if entry.get("reference_text"):
                bullet_lines.append(f"- 参考文献: {_md_value(entry.get('reference_text'))}")
            if entry.get("corrected_citation"):
                bullet_lines.append(f"- 建议修正: {_md_value(entry.get('corrected_citation'))}")
            if entry.get("formatted_citation"):
                bullet_lines.append(f"- 建议格式化: {_md_value(entry.get('formatted_citation'))}")
            if entry.get("formatting_reason"):
                bullet_lines.append(f"- 原因: {_md_value(entry.get('formatting_reason'))}")
            lines.extend(bullet_lines)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_markdown_report(report: Dict[str, Any], output_path: str) -> str:
    """
    将报告保存为 Markdown 文件。

    Args:
        report: PaperChecker JSON 字典。
        output_path: 输出文件路径。

    Returns:
        输出文件路径。
    """
    content = build_markdown_report(report)
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(content)
    return output_path


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="将 PaperChecker JSON 报告转换为 Markdown"
    )
    parser.add_argument("--json", required=True, help="JSON 报告路径")
    parser.add_argument("--output", required=True, help="Markdown 输出路径")
    args = parser.parse_args()

    json_path = Path(args.json)
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    save_markdown_report(report, args.output)


if __name__ == "__main__":
    main()
