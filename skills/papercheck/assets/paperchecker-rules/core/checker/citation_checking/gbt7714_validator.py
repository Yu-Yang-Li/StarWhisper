"""
GB/T 7714 基础规则校验（轻量实现）

目标：
1. 不改动核心匹配流程的前提下，提供可复用的著录规则检测能力。
2. 重点覆盖高频且高价值问题（P0/P1）：
   - 正文引文体系与文后组织方式的基本一致性
   - 参考文献条目必要要素缺失
   - 报纸 [N] 日期/版次完整性
   - 数字制编号起始、连续性、重复
"""

from typing import Dict, List, Optional
import re


_REFERENCE_NO_RE = re.compile(r"^\s*[\[［〔【](\d+)[\]］〕】]")
_DOC_TYPE_RE = re.compile(r"[\[［]\s*([A-Za-z]+(?:/[A-Za-z]+)?)\s*[\]］]")
_YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_NEWS_DATE_RE = re.compile(r"(?<!\d)(?:18|19|20)\d{2}-\d{2}-\d{2}(?!\d)")
_NEWS_EDITION_RE = re.compile(r"[（(]\s*[0-9０-９一二三四五六七八九十]+\s*[）)]")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_ACCESS_DATE_RE = re.compile(r"[\[［]\s*(?:18|19|20)\d{2}-\d{2}-\d{2}\s*[\]］]")
_ONLINE_PUBLISH_DATE_RE = re.compile(r"[（(]\s*(?:18|19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?\s*[）)]")
_STANDARD_CODE_RE = re.compile(
    r"\b(?:GB(?:/T)?|ISO|IEC|EN|ASTM|BS|DIN|DB\d*|T/[A-Z0-9]+)"
    r"\s*[A-Z0-9.\-]*\d+(?:[:\-]\d{4})?\b",
    re.IGNORECASE,
)
_PATENT_NO_RE = re.compile(r"\b(?:[A-Z]{2}\d{6,}|\d{8,}(?:\.\d+)?)\b", re.IGNORECASE)
_JOURNAL_LOCATOR_RE = re.compile(r"[:：]\s*(?:[Ee]?\d+(?:\s*[-–]\s*\d+)?|[A-Za-z]{1,3}\d{3,})")
_DOI_RE = re.compile(r"\bdoi\s*[:：]?\s*10\.\d{4,9}/\S+", re.IGNORECASE)
_BOOK_PUBLISHER_RE = re.compile(
    r"(?:[:：]\s*[^。]+[,，]\s*(?:18|19|20)\d{2}|(?:出版社|press|publisher)[^,，。.]*[,，]\s*(?:18|19|20)\d{2})",
    re.IGNORECASE,
)
_REPORT_NUMBER_RE = re.compile(
    r"(?:report\s*no\.?\s*[\w\-./]+|working\s*paper(?:\s*no\.?)?\s*[\w\-./]+|"
    r"报告编号[:：]?\s*[\w\-./]+|编号[:：]?\s*[\w\-./]+|第\s*[\w一二三四五六七八九十百千]+\s*号)",
    re.IGNORECASE,
)
_REPORT_PAGE_LOCATOR_RE = re.compile(
    r"(?:"
    r"(?:18|19|20)\d{2}\s*[:：]\s*[A-Za-z]?\d+(?:\s*[-–]\s*[A-Za-z]?\d+)?"
    r"|pp?\.?\s*\d+(?:\s*[-–]\s*\d+)?"
    r"|\d+\s*[-–]\s*\d+\s*页"
    r"|\d+\s*页"
    r")",
    re.IGNORECASE,
)
_THESIS_SOURCE_RE = re.compile(
    r"(大学|学院|学位论文|thesis|dissertation|university|college)",
    re.IGNORECASE,
)


def _append_issue(
    issues: List[Dict[str, str]],
    issue_code: str,
    severity: str,
    reference_text: str,
    message: str,
    suggestion: str,
) -> None:
    issues.append({
        "issue_code": issue_code,
        "severity": severity,
        "reference_text": reference_text,
        "message": message,
        "suggestion": suggestion,
    })


def extract_reference_no(reference_text: str) -> Optional[int]:
    """提取参考文献编号（如 [12]）。"""
    if not reference_text:
        return None
    match = _REFERENCE_NO_RE.match(reference_text.strip())
    if not match:
        return None
    return int(match.group(1))


def extract_doc_type_marker(reference_text: str) -> Optional[str]:
    """
    提取文献类型标识（如 [J]/[N]/[EB/OL]）。
    会跳过条目前缀编号 [1]。
    """
    if not reference_text:
        return None

    for match in _DOC_TYPE_RE.finditer(reference_text):
        marker = match.group(1).upper()
        if marker.isdigit():
            continue
        return marker
    return None


def validate_reference_entry(reference_text: str, citation_style: str) -> List[Dict[str, str]]:
    """校验单条参考文献，返回结构化问题列表。"""
    issues: List[Dict[str, str]] = []
    ref_text = (reference_text or "").strip()
    if not ref_text:
        return issues

    ref_no = extract_reference_no(ref_text)
    doc_type = extract_doc_type_marker(ref_text)
    doc_type_base = (doc_type.split("/", 1)[0] if doc_type else "")
    online_carrier = bool(doc_type and doc_type.upper().endswith("/OL"))

    if citation_style == "numeric" and ref_no is None:
        _append_issue(
            issues,
            "REF_NUMERIC_MISSING_INDEX",
            "error",
            ref_text,
            "顺序编码制下，参考文献条目应以 [n] 编号起始。",
            "为该条参考文献补充顺序编号（如 [31] ...）。",
        )

    if citation_style == "author_year" and ref_no is not None:
        _append_issue(
            issues,
            "REF_AUTHOR_YEAR_HAS_INDEX",
            "warn",
            ref_text,
            "著者-出版年制下，文后参考文献通常不加 [n] 编号。",
            "如整篇采用著者-出版年制，建议去掉文后条目编号。",
        )

    if not doc_type:
        _append_issue(
            issues,
            "REF_MISSING_DOC_TYPE",
            "error",
            ref_text,
            "未识别到文献类型标识（如 [J]/[M]/[N]/[D]/[S]/[P]/[EB/OL]）。",
            "按 GB/T 7714 在题名后补充文献类型标识。",
        )
        return issues

    # 需要年份的常见文献类型
    year_required_types = {"J", "M", "D", "R", "S", "P", "C", "A", "Z", "N"}
    if (doc_type in {"EB/OL"} or doc_type_base in year_required_types) and not _YEAR_RE.search(ref_text):
        _append_issue(
            issues,
            "REF_MISSING_YEAR",
            "error",
            ref_text,
            f"{doc_type} 类型文献未识别到有效出版年份（18xx/19xx/20xx）。",
            "补充 4 位出版年份。",
        )

    # 期刊 [J]：通常应有页码或文章编号信息
    if doc_type_base == "J" and not (_JOURNAL_LOCATOR_RE.search(ref_text) or _DOI_RE.search(ref_text)):
        _append_issue(
            issues,
            "REF_J_MISSING_PAGES_OR_ARTICLE_NO",
            "warn",
            ref_text,
            "期刊文献 [J] 未识别到页码/文章编号信息（如 ': 1-10'）。",
            "补充页码范围或文章编号。",
        )

    # 专著 [M]：通常应包含出版项（出版地: 出版者, 年）
    if doc_type_base == "M" and not _BOOK_PUBLISHER_RE.search(ref_text):
        _append_issue(
            issues,
            "REF_M_MISSING_PUBLISHER_INFO",
            "warn",
            ref_text,
            "专著 [M] 未识别到完整出版项（出版地: 出版者, 年）。",
            "补充出版地、出版社和出版年。",
        )

    # 学位论文 [D]：通常应体现授予单位/学位论文属性
    if doc_type_base == "D" and not _THESIS_SOURCE_RE.search(ref_text):
        _append_issue(
            issues,
            "REF_D_MISSING_THESIS_SOURCE",
            "warn",
            ref_text,
            "学位论文 [D] 未识别到授予单位或学位论文来源信息。",
            "补充授予单位（如某大学）或学位论文来源描述。",
        )

    # 标准 [S]：应有标准编号
    if doc_type_base == "S" and not _STANDARD_CODE_RE.search(ref_text):
        _append_issue(
            issues,
            "REF_S_MISSING_STANDARD_CODE",
            "error",
            ref_text,
            "标准文献 [S] 未识别到标准编号（如 GB/T 7714-2015）。",
            "补充标准编号。",
        )

    # 专利 [P]：应有专利号
    if doc_type_base == "P" and not _PATENT_NO_RE.search(ref_text):
        _append_issue(
            issues,
            "REF_P_MISSING_PATENT_NO",
            "error",
            ref_text,
            "专利文献 [P] 未识别到专利号。",
            "补充专利号。",
        )

    # 析出文献 [A]：通常应出现 // 连接母体文献
    if doc_type_base == "A" and "//" not in ref_text:
        _append_issue(
            issues,
            "REF_A_MISSING_IN_CONTAINER_MARK",
            "warn",
            ref_text,
            "析出文献 [A] 未识别到 '//' 母体文献连接符。",
            "补充 '//' 并著录母体文献信息。",
        )

    # 报纸文献 [N]：要求出版日期 YYYY-MM-DD，且通常带版次
    if doc_type_base == "N":
        if not _NEWS_DATE_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_NEWS_MISSING_FULL_DATE",
                "error",
                ref_text,
                "报纸文献 [N] 的出版日期不完整，需为 YYYY-MM-DD。",
                "将日期补全为 YYYY-MM-DD，例如 2025-05-20。",
            )
        if not _NEWS_EDITION_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_NEWS_MISSING_EDITION",
                "warn",
                ref_text,
                "报纸文献 [N] 未识别到版次信息（如（1））。",
                "补充版次信息，例如（1）。",
            )

    # 报告文献 [R]：重点校验出版项、报告编号、页码定位
    if doc_type_base == "R":
        if not online_carrier and not _BOOK_PUBLISHER_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_R_MISSING_PUBLISHER_INFO",
                "error",
                ref_text,
                "报告文献 [R] 未识别到完整出版项（出版地: 发布机构, 年）。",
                "按 GB/T 7714 补充出版地、发布机构与年份。",
            )
        if not _REPORT_NUMBER_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_R_MISSING_REPORT_NO",
                "warn",
                ref_text,
                "报告文献 [R] 未识别到报告编号（如 Report No./报告编号）。",
                "建议补充报告编号，提升可追溯性。",
            )
        if not _REPORT_PAGE_LOCATOR_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_R_MISSING_PAGE_LOCATOR",
                "warn",
                ref_text,
                "报告文献 [R] 未识别到引文页码定位信息（如 2020: 12-18 或 pp.12-18）。",
                "建议补充引文页码定位。",
            )

    # 电子载体文献（如 [EB/OL]、[R/OL]）：建议含 URL 与引用日期
    if doc_type == "EB/OL" or online_carrier:
        if not _URL_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_EBOL_MISSING_URL",
                "warn",
                ref_text,
                "电子文献 [EB/OL] 未识别到访问路径（URL）。",
                "补充可访问的 URL。",
            )
        if not _ACCESS_DATE_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_EBOL_MISSING_ACCESS_DATE",
                "warn",
                ref_text,
                "电子文献 [EB/OL] 未识别到引用日期（如 [2026-04-06]）。",
                "补充引用日期 [YYYY-MM-DD]。",
            )
        if doc_type_base == "R" and not _ONLINE_PUBLISH_DATE_RE.search(ref_text):
            _append_issue(
                issues,
                "REF_R_OL_MISSING_PUBLISH_DATE",
                "warn",
                ref_text,
                "在线报告 [R/OL] 未识别到发布时间（如 (2013-04-16)）。",
                "补充发布时间（YYYY 或 YYYY-MM-DD）并置于括号中。",
            )

    return issues


def validate_numeric_reference_numbering(reference_texts: List[str]) -> List[Dict[str, str]]:
    """校验顺序编码制参考文献编号的起始、连续性与重复。"""
    issues: List[Dict[str, str]] = []
    numbers: List[int] = []
    number_to_refs: Dict[int, List[str]] = {}

    for text in reference_texts:
        number = extract_reference_no(text)
        if number is None:
            continue
        numbers.append(number)
        number_to_refs.setdefault(number, []).append((text or "").strip())

    if not numbers:
        return issues

    min_no = min(numbers)
    max_no = max(numbers)

    if min_no != 1:
        _append_issue(
            issues,
            "REF_NUMERIC_START_NOT_ONE",
            "warn",
            f"编号最小值为 [{min_no}]",
            "顺序编码制下，参考文献编号通常应从 [1] 开始。",
            "检查文后参考文献编号起始值并统一调整。",
        )

    unique_numbers = set(numbers)
    missing = [n for n in range(min_no, max_no + 1) if n not in unique_numbers]
    if missing:
        preview = ", ".join(str(n) for n in missing[:10])
        if len(missing) > 10:
            preview += ", ..."
        _append_issue(
            issues,
            "REF_NUMERIC_MISSING_SEQUENCE",
            "error",
            f"缺失编号: {preview}",
            "顺序编码制参考文献编号存在断档（不连续）。",
            "补齐或重排缺失编号，保持连续。",
        )

    duplicates = sorted(n for n, refs in number_to_refs.items() if len(refs) > 1)
    for dup_no in duplicates:
        _append_issue(
            issues,
            "REF_NUMERIC_DUPLICATE_INDEX",
            "error",
            f"[{dup_no}] " + " | ".join(number_to_refs[dup_no][:2]),
            f"发现重复编号 [{dup_no}]。",
            "确保每个参考文献编号唯一。",
        )

    return issues
