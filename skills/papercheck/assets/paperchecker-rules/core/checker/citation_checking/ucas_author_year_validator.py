from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from models.document import Document, Reference
from core.checker.citation_checking.gbt7714_validator import extract_doc_type_marker


YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})([a-z]?)(?!\d)", re.IGNORECASE)
PAREN_GROUP_RE = re.compile(r"[（(]([^()（）]{1,200})[）)]")
INLINE_AUTHOR_YEAR_RE = re.compile(
    r"(?P<author>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z·\s\.\-&]{0,80})\s*[（(](?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)[）)]",
    re.IGNORECASE,
)
SEGMENT_AUTHOR_YEAR_RE = re.compile(
    r"^\s*(?P<author>[\u4e00-\u9fffA-Za-zÀ-ÖØ-öø-ÿ][\u4e00-\u9fffA-Za-zÀ-ÖØ-öø-ÿ·\s\.\-&'’]{0,100}?)\s*[,，]\s*"
    r"(?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)(?:\s*(?P<page_sep>[,，:：])\s*(?P<page>\d+(?:\s*[-–]\s*\d+)?))?\s*$",
    re.IGNORECASE,
)
REFERENCE_INDEX_RE = re.compile(r"^\s*[\[［〔【]\d+[\]］〕】]\s*")
NON_ARABIC_NUM_RE = re.compile(r"[一二三四五六七八九十百千〇零壹贰叁肆伍陆柒捌玖拾两]")
NON_GREGORIAN_RE = re.compile(r"(民国|昭和|平成|康熙|乾隆|天保|大正|元年)")
PAREN_NON_GREGORIAN_RE = re.compile(r"[（(][^()（）]*(民国|昭和|平成|康熙|乾隆|天保|大正|元年)[^()（）]*[）)]")
DUP_PUNCT_RE = re.compile(r"(,,|，，|::|：：|\.\.|。。|;;|；；|、、)")

RULE_STRENGTH_STRONG = "strong_rule"
RULE_STRENGTH_HEURISTIC = "heuristic_rule"
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

UCAS_ISSUE_CLASSIFICATION: Dict[str, Dict[str, Any]] = {
    # P2: 启发式规则（保留但降低置信度）
    "UCAS_NOTE_ORG_AUTHOR_FULLNAME": {
        "rule_strength": RULE_STRENGTH_HEURISTIC,
        "confidence_tier": CONFIDENCE_LOW,
        "confidence_score": 0.58,
    },
    "UCAS_NOTE_WESTERN_NAME_ORDER": {
        "rule_strength": RULE_STRENGTH_HEURISTIC,
        "confidence_tier": CONFIDENCE_MEDIUM,
        "confidence_score": 0.62,
    },
    "UCAS_NOTE_FOREIGN_TITLE_CAPITALIZATION": {
        "rule_strength": RULE_STRENGTH_HEURISTIC,
        "confidence_tier": CONFIDENCE_MEDIUM,
        "confidence_score": 0.66,
    },
    "UCAS_NOTE_FOREIGN_JOURNAL_FULLNAME": {
        "rule_strength": RULE_STRENGTH_HEURISTIC,
        "confidence_tier": CONFIDENCE_MEDIUM,
        "confidence_score": 0.65,
    },
    # 缩写点规则可自动识别，但存在文种边界差异，设为中置信
    "UCAS_NOTE_WESTERN_INITIAL_NO_DOT": {
        "rule_strength": RULE_STRENGTH_STRONG,
        "confidence_tier": CONFIDENCE_MEDIUM,
        "confidence_score": 0.82,
    },
}


@dataclass
class CitationMention:
    author: str
    author_key: str
    year: int
    suffix: str
    raw_text: str


def validate_ucas_author_year_body(document: Document) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    seen = set()
    body_text = _join_body_text(document.content or [])

    mentions: List[CitationMention] = []

    for group_match in PAREN_GROUP_RE.finditer(body_text):
        group_text = group_match.group(1).strip()
        if not YEAR_RE.search(group_text):
            continue

        segments = [seg.strip() for seg in re.split(r"[;；]", group_text) if seg.strip()]
        if len(segments) > 1:
            normalized = "; ".join(segments)
            if group_text != normalized:
                _append_issue_once(
                    issues,
                    seen,
                    (
                        "UCAS_BODY_MULTI_CITATION_SEPARATOR",
                        group_text,
                    ),
                    issue_code="UCAS_BODY_MULTI_CITATION_SEPARATOR",
                    severity="warn",
                    citation_text=f"（{group_text}）",
                    message="同处多文献著者-出版年标注应使用半角分号并在分号后保留一个空格。",
                    suggestion="将同处多引文统一写为“(Author, 2001; Author, 2005)”格式。",
                )

            year_seq = [_extract_first_year(segment) for segment in segments]
            numeric_year_seq = [year for year in year_seq if year is not None]
            if len(numeric_year_seq) >= 2 and numeric_year_seq != sorted(numeric_year_seq):
                _append_issue_once(
                    issues,
                    seen,
                    (
                        "UCAS_BODY_MULTI_CITATION_YEAR_ORDER",
                        group_text,
                    ),
                    issue_code="UCAS_BODY_MULTI_CITATION_YEAR_ORDER",
                    severity="warn",
                    citation_text=f"（{group_text}）",
                    message="同处多文献著者-出版年标注应按年份由远及近排序。",
                    suggestion="将同处多引文按年份升序排列。",
                )

        for segment in segments:
            parsed = _parse_parenthetical_segment(segment)
            if parsed is None:
                continue

            mention = CitationMention(
                author=parsed["author"],
                author_key=_normalize_author_key(parsed["author"]),
                year=parsed["year"],
                suffix=parsed["suffix"],
                raw_text=f"（{segment}）",
            )
            mentions.append(mention)

            _validate_chinese_author_rule(parsed, issues, seen, mention.raw_text)
            _validate_western_author_rule(parsed, issues, seen, mention.raw_text)

            if _has_inline_page_locator(segment, parsed["year"], parsed.get("suffix", "")):
                _append_issue_once(
                    issues,
                    seen,
                    (
                        "UCAS_BODY_PAGE_POSITION",
                        segment,
                    ),
                    issue_code="UCAS_BODY_PAGE_POSITION",
                    severity="warn",
                    citation_text=mention.raw_text,
                    message="同一文献多次引用时，页码应置于括号外（如“(张三, 2020)8”）。",
                    suggestion="将页码移至右括号外侧，示例“(张三, 2020)12-15”。",
                )

    for inline_match in INLINE_AUTHOR_YEAR_RE.finditer(body_text):
        author = (inline_match.group("author") or "").strip()
        year = int(inline_match.group("year"))
        suffix = (inline_match.group("suffix") or "").lower()
        raw = inline_match.group(0)
        mention = CitationMention(
            author=author,
            author_key=_normalize_author_key(author),
            year=year,
            suffix=suffix,
            raw_text=raw,
        )
        mentions.append(mention)

        parsed = {
            "author": author,
            "year": year,
            "suffix": suffix,
            "raw_text": raw,
        }
        _validate_chinese_author_rule(parsed, issues, seen, raw)
        _validate_western_author_rule(parsed, issues, seen, raw)

    _validate_same_author_same_year_suffix(document.references or [], mentions, issues, seen)
    return issues


def validate_ucas_reference_order(reference_items: List[Reference]) -> List[Dict[str, str]]:
    parsed_items: List[Optional[Dict[str, Any]]] = []
    for idx, item in enumerate(reference_items):
        parsed_items.append(_parse_reference_sort_key(item.text, idx))

    issues: List[Dict[str, str]] = []
    seen = set()
    for idx in range(len(parsed_items) - 1):
        current = parsed_items[idx]
        nxt = parsed_items[idx + 1]
        if current is None or nxt is None:
            continue

        if _sort_tuple(current) <= _sort_tuple(nxt):
            continue

        if current["lang_rank"] > nxt["lang_rank"]:
            code = "UCAS_REF_LANGUAGE_ORDER"
            message = "著者-出版年制下，文后参考文献应按语种顺序排列（中文→日文→西文→俄文→其他）。"
            suggestion = "调整该处语种顺序，确保中文条目在日文、西文、俄文之前。"
        elif current["author_key"] > nxt["author_key"]:
            code = "UCAS_REF_AUTHOR_ORDER"
            message = "同语种下文后参考文献应按第一作者字序排序。"
            suggestion = "按第一作者字序重排相邻条目。"
        elif current["author_key"] == nxt["author_key"] and current["single_rank"] > nxt["single_rank"]:
            code = "UCAS_REF_SINGLE_BEFORE_COAUTHOR"
            message = "同一第一作者条目应先列单独署名，再列合著文献。"
            suggestion = "将同一作者的单著文献移到合著文献之前。"
        elif current["author_key"] == nxt["author_key"] and current["year"] > nxt["year"]:
            code = "UCAS_REF_YEAR_ORDER"
            message = "同一作者条目应按出版年升序排列。"
            suggestion = "将同一作者条目按年份由早到晚重排。"
        else:
            code = "UCAS_REF_ORDER"
            message = "著者-出版年制文后参考文献排序不符合 UCAS 要求。"
            suggestion = "按语种、作者字序、作者人数与年份规则重排。"

        pair_text = f"{current['text']} || {nxt['text']}"
        _append_issue_once(
            issues,
            seen,
            (code, pair_text),
            issue_code=code,
            severity="warn",
            reference_text=pair_text,
            message=message,
            suggestion=suggestion,
        )

    return issues


def validate_ucas_reference_notes(reference_items: List[Reference]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    seen = set()

    for item in reference_items:
        raw = (item.text or "").strip()
        if not raw:
            continue

        ref_text = REFERENCE_INDEX_RE.sub("", raw).strip()
        doc_type = (extract_doc_type_marker(raw) or "").upper()
        doc_type_base = doc_type.split("/", 1)[0] if doc_type else ""

        author_part, title_part, rest_part = _split_reference_core(ref_text)
        latin_author = _contains_latin(author_part) and not _contains_cjk(author_part)
        cjk_author = _contains_cjk(author_part)

        if not author_part:
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_UNKNOWN_AUTHOR_NOT_ANON", raw),
                issue_code="UCAS_NOTE_UNKNOWN_AUTHOR_NOT_ANON",
                severity="warn",
                reference_text=raw,
                message="作者不明文献应在责任者位置注明“佚名”或对应词。",
                suggestion="将缺失作者改为“佚名”或“Anonymous”等对应标识。",
            )

        if _is_org_abbreviation(author_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_ORG_AUTHOR_FULLNAME", raw),
                issue_code="UCAS_NOTE_ORG_AUTHOR_FULLNAME",
                severity="warn",
                reference_text=raw,
                message="机构/团体作者建议使用全称，避免仅使用缩写。",
                suggestion="将机构简称改为机构全称。",
            )

        explicit_count = _explicit_author_count(author_part)
        has_etal = bool(re.search(r"\bet\s+al\.?\b", author_part, flags=re.IGNORECASE) or "等" in author_part)
        if has_etal and explicit_count < 3:
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_AUTHOR_COUNT_LE3_NO_ETAL", raw),
                issue_code="UCAS_NOTE_AUTHOR_COUNT_LE3_NO_ETAL",
                severity="warn",
                reference_text=raw,
                message="使用“等 / et al.”时，建议至少著录前 3 位作者。",
                suggestion="将作者调整为“前3名作者 + 等 / et al.”。",
            )
        if explicit_count > 3 and not has_etal:
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_AUTHOR_COUNT_GT3_REQUIRE_ETAL", raw),
                issue_code="UCAS_NOTE_AUTHOR_COUNT_GT3_REQUIRE_ETAL",
                severity="warn",
                reference_text=raw,
                message="作者人数超过 3 人时，建议仅著录前 3 人并加“，等”或“, et al.”。",
                suggestion="改为“前3名作者 + 等 / et al.”格式。",
            )

        if cjk_author and "等" in author_part and not re.search(r"[，,]\s*等", author_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_CN_ETAL_PUNCT", raw),
                issue_code="UCAS_NOTE_CN_ETAL_PUNCT",
                severity="warn",
                reference_text=raw,
                message="中文作者截断格式建议使用“，等”。",
                suggestion="将作者截断写为“前3名作者，等”。",
            )

        if latin_author and re.search(r"\bet\s+al\b", author_part, flags=re.IGNORECASE) and not re.search(
            r",\s*et\s+al\.?",
            author_part,
            flags=re.IGNORECASE,
        ):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_EN_ETAL_PUNCT", raw),
                issue_code="UCAS_NOTE_EN_ETAL_PUNCT",
                severity="warn",
                reference_text=raw,
                message="英文作者截断格式建议使用“, et al.”。",
                suggestion="将作者截断写为“Author1, Author2, Author3, et al.”。",
            )

        if latin_author and re.search(r"\b[A-Z]\.\b|\b[A-Z]\.", author_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_WESTERN_INITIAL_NO_DOT", raw),
                issue_code="UCAS_NOTE_WESTERN_INITIAL_NO_DOT",
                severity="warn",
                reference_text=raw,
                message="西文作者名字缩写建议不加缩写点。",
                suggestion="将“J.”改为“J”等不带缩写点形式。",
            )

        if latin_author and _looks_given_name_first(author_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_WESTERN_NAME_ORDER", raw),
                issue_code="UCAS_NOTE_WESTERN_NAME_ORDER",
                severity="warn",
                reference_text=raw,
                message="西文作者名录建议采用“姓在前，名在后（或名缩写）”格式。",
                suggestion="将作者格式调整为“Surname Given-initials”。",
            )

        if _contains_latin(title_part) and _is_foreign_title_all_caps(title_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_FOREIGN_TITLE_CAPITALIZATION", raw),
                issue_code="UCAS_NOTE_FOREIGN_TITLE_CAPITALIZATION",
                severity="warn",
                reference_text=raw,
                message="外文题名建议遵循文种本身的大小写习惯，不应整体大写。",
                suggestion="按原文习惯调整外文题名大小写。",
            )

        if doc_type_base == "J":
            journal_name = _extract_journal_name(ref_text)
            if _looks_abbreviated_journal_name(journal_name):
                _append_issue_once(
                    issues,
                    seen,
                    ("UCAS_NOTE_FOREIGN_JOURNAL_FULLNAME", raw),
                    issue_code="UCAS_NOTE_FOREIGN_JOURNAL_FULLNAME",
                    severity="warn",
                    reference_text=raw,
                    message="外文期刊名建议著录全称，不建议使用缩写。",
                    suggestion="将外文期刊名替换为全称。",
                )

            if not _has_journal_min_fields(ref_text):
                _append_issue_once(
                    issues,
                    seen,
                    ("UCAS_NOTE_JOURNAL_MIN_FIELDS", raw),
                    issue_code="UCAS_NOTE_JOURNAL_MIN_FIELDS",
                    severity="error",
                    reference_text=raw,
                    message="期刊条目应包含“年, 卷(期): 起止页”，且卷或期至少一项。",
                    suggestion="补充年、卷(期)与页码信息。",
                )

        if _has_non_arabic_numerals_for_biblio_fields(rest_part):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_NON_ARABIC_NUMERALS", raw),
                issue_code="UCAS_NOTE_NON_ARABIC_NUMERALS",
                severity="warn",
                reference_text=raw,
                message="版次、卷、期、页码等建议统一使用阿拉伯数字。",
                suggestion="将中文数字改为阿拉伯数字形式（如“第3版”）。",
            )

        if NON_GREGORIAN_RE.search(ref_text) and not PAREN_NON_GREGORIAN_RE.search(ref_text):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_NON_GREGORIAN_PARENTHESES", raw),
                issue_code="UCAS_NOTE_NON_GREGORIAN_PARENTHESES",
                severity="warn",
                reference_text=raw,
                message="非公元纪年应置于括号中著录。",
                suggestion="改为“YYYY（非公元纪年）”格式。",
            )

        if ref_text.count("[") != ref_text.count("]") or ref_text.count("（") != ref_text.count("）") or ref_text.count("(") != ref_text.count(")"):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_PUNCT_BRACKET_BALANCE", raw),
                issue_code="UCAS_NOTE_PUNCT_BRACKET_BALANCE",
                severity="warn",
                reference_text=raw,
                message="参考文献中的括号/方括号未成对，标点格式可能不规范。",
                suggestion="检查并补全成对括号与方括号。",
            )

        if DUP_PUNCT_RE.search(ref_text):
            _append_issue_once(
                issues,
                seen,
                ("UCAS_NOTE_PUNCT_DUPLICATE", raw),
                issue_code="UCAS_NOTE_PUNCT_DUPLICATE",
                severity="warn",
                reference_text=raw,
                message="参考文献中存在重复标点，建议按规范统一标点。",
                suggestion="去除重复标点并按 GB/T 7714 规范调整。",
            )

    return issues


def _append_issue_once(
    issues: List[Dict[str, Any]],
    seen: set,
    key: Tuple[str, str],
    *,
    issue_code: str,
    severity: str,
    message: str,
    suggestion: str,
    citation_text: Optional[str] = None,
    reference_text: Optional[str] = None,
    **extra_fields: Any,
) -> None:
    if key in seen:
        return
    seen.add(key)
    classification = _get_issue_classification(issue_code)
    payload = {
        "issue_code": issue_code,
        "severity": severity,
        "message": message,
        "suggestion": suggestion,
        "rule_strength": classification["rule_strength"],
        "confidence_tier": classification["confidence_tier"],
        "confidence_score": classification["confidence_score"],
    }
    if citation_text is not None:
        payload["citation_text"] = citation_text
    if reference_text is not None:
        payload["reference_text"] = reference_text
    if extra_fields:
        payload.update(extra_fields)
    issues.append(payload)


def _get_issue_classification(issue_code: str) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "rule_strength": RULE_STRENGTH_STRONG,
        "confidence_tier": CONFIDENCE_HIGH,
        "confidence_score": 0.95,
    }
    override = UCAS_ISSUE_CLASSIFICATION.get(issue_code)
    if not override:
        return default
    merged = dict(default)
    merged.update(override)
    return merged


def _join_body_text(paragraphs: List[str]) -> str:
    if not paragraphs:
        return ""
    ref_idx = -1
    for idx, raw in enumerate(paragraphs):
        text = (raw or "").strip().lower()
        if text in {"参考文献", "references"}:
            ref_idx = idx
    body = paragraphs[:ref_idx] if ref_idx != -1 else paragraphs
    return "\n".join(str(item or "") for item in body)


def _parse_parenthetical_segment(segment: str) -> Optional[Dict[str, Any]]:
    match = SEGMENT_AUTHOR_YEAR_RE.search(segment)
    if match:
        return {
            "author": (match.group("author") or "").strip(),
            "year": int(match.group("year")),
            "suffix": (match.group("suffix") or "").lower(),
            "has_inline_page": bool(match.group("page")),
            "raw_text": segment,
        }

    # 兼容西文多作者逗号并列：Brown, Clark, Davis, 2022
    western_multi = re.search(
        r"(?P<author>[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s\.\-&'’]{0,80}(?:\s*,\s*[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s\.\-&'’]{0,80}){1,5})\s*,\s*"
        r"(?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)(?:\s*(?P<page_sep>[,，:：])\s*(?P<page>\d+(?:\s*[-–]\s*\d+)?))?\s*$",
        segment,
        flags=re.IGNORECASE,
    )
    if western_multi:
        return {
            "author": (western_multi.group("author") or "").strip(),
            "year": int(western_multi.group("year")),
            "suffix": (western_multi.group("suffix") or "").lower(),
            "has_inline_page": bool(western_multi.group("page")),
            "raw_text": segment,
        }

    # 允许 Author（2020）形式落入同一校验
    inline = INLINE_AUTHOR_YEAR_RE.search(segment)
    if inline:
        return {
            "author": (inline.group("author") or "").strip(),
            "year": int(inline.group("year")),
            "suffix": (inline.group("suffix") or "").lower(),
            "raw_text": segment,
        }
    return None


def _validate_chinese_author_rule(
    parsed: Dict[str, Any],
    issues: List[Dict[str, str]],
    seen: set,
    raw_text: str,
) -> None:
    author = parsed.get("author", "")
    if not _contains_cjk(author):
        return
    if not _should_apply_cn_person_name_rules(author):
        return

    year = parsed.get("year")
    suffix = (parsed.get("suffix") or "").lower()
    display_author = _normalize_chinese_author_display(author)
    normalized_citation = _build_cn_author_year_citation(display_author, year, suffix)
    first_author = _extract_first_cn_author(display_author)

    if _looks_multi_author(author) and "等" not in author:
        suggested_citation = _build_cn_author_year_citation(f"{first_author} 等", year, suffix)
        _append_issue_once(
            issues,
            seen,
            ("UCAS_BODY_CN_MULTI_AUTHOR_ET_AL", raw_text),
            issue_code="UCAS_BODY_CN_MULTI_AUTHOR_ET_AL",
            severity="error",
            citation_text=normalized_citation,
            citation_normalized=normalized_citation,
            suggested_citation=suggested_citation,
            message="中文多作者正文标注应使用“第一作者 + 空格 + 等”。",
            suggestion=f"建议改为“{suggested_citation}”。",
        )
        return

    if "等" in author and not re.search(r"\S\s等$", author.strip()):
        original_author = re.sub(r"\s*等$", "等", display_author)
        suggested_author = re.sub(r"\s*等$", " 等", display_author)
        original_citation = _build_cn_author_year_citation(original_author, year, suffix)
        suggested_citation = _build_cn_author_year_citation(suggested_author, year, suffix)
        _append_issue_once(
            issues,
            seen,
            ("UCAS_BODY_CN_ET_SPACE", raw_text),
            issue_code="UCAS_BODY_CN_ET_SPACE",
            severity="error",
            citation_text=original_citation,
            citation_normalized=original_citation,
            suggested_citation=suggested_citation,
            message="中文多作者“等”前应保留一个空格。",
            suggestion=f"将“{original_author}”改为“{suggested_author}”。",
        )


def _validate_western_author_rule(
    parsed: Dict[str, Any],
    issues: List[Dict[str, str]],
    seen: set,
    raw_text: str,
) -> None:
    author = parsed.get("author", "")
    if _contains_cjk(author) or not _contains_latin(author):
        return

    lowered = author.lower()
    if "et al." in lowered:
        return

    author_count = _estimate_western_author_count(author)
    if author_count >= 3:
        _append_issue_once(
            issues,
            seen,
            ("UCAS_BODY_WESTERN_THREE_PLUS_ETAL", raw_text),
            issue_code="UCAS_BODY_WESTERN_THREE_PLUS_ETAL",
            severity="error",
            citation_text=raw_text,
            message="西文 3 位及以上作者应写作“第一作者 et al., 年份”。",
            suggestion="将多作者西文引文改为“Surname et al., YYYY”。",
        )
    elif author_count == 2 and "&" not in author:
        _append_issue_once(
            issues,
            seen,
            ("UCAS_BODY_WESTERN_TWO_AUTHOR_AMPERSAND", raw_text),
            issue_code="UCAS_BODY_WESTERN_TWO_AUTHOR_AMPERSAND",
            severity="error",
            citation_text=raw_text,
            message="西文 2 位作者应使用“&”连接。",
            suggestion="将“A and B”改为“A & B”。",
        )


def _validate_same_author_same_year_suffix(
    references: List[Reference],
    mentions: List[CitationMention],
    issues: List[Dict[str, str]],
    seen: set,
) -> None:
    ref_counter: Dict[Tuple[str, int], int] = {}
    for ref in references:
        parsed = _parse_reference_sort_key(ref.text, 0)
        if parsed is None or parsed["year"] <= 0:
            continue
        key = (parsed["author_key"], parsed["year"])
        ref_counter[key] = ref_counter.get(key, 0) + 1

    duplicate_keys = {key for key, count in ref_counter.items() if count > 1}
    if not duplicate_keys:
        return

    for mention in mentions:
        key = (mention.author_key, mention.year)
        if key not in duplicate_keys:
            continue
        if mention.suffix:
            continue
        _append_issue_once(
            issues,
            seen,
            ("UCAS_BODY_SAME_AUTHOR_YEAR_SUFFIX", mention.raw_text),
            issue_code="UCAS_BODY_SAME_AUTHOR_YEAR_SUFFIX",
            severity="error",
            citation_text=mention.raw_text,
            message="同一作者同年多条文献应使用 a/b/c 后缀区分。",
            suggestion="将年份改为带后缀格式，如 2020a、2020b。",
        )


def _parse_reference_sort_key(reference_text: str, index: int) -> Optional[Dict[str, Any]]:
    raw = (reference_text or "").strip()
    if not raw:
        return None

    normalized = REFERENCE_INDEX_RE.sub("", raw)
    author_part = normalized
    marker_pos = normalized.find("[")
    if marker_pos > 0:
        author_part = normalized[:marker_pos]
    author_part = author_part.strip()

    first_author = _extract_first_author(author_part)
    if not first_author:
        first_author = author_part[:20].strip()

    year = _extract_first_year(raw) or 0
    lang_rank = _language_rank(first_author)
    author_key = _normalize_author_key(first_author)
    single_rank = 0 if _estimate_author_count(author_part) <= 1 else 1

    return {
        "index": index,
        "text": raw,
        "first_author": first_author,
        "author_key": author_key,
        "year": year,
        "lang_rank": lang_rank,
        "single_rank": single_rank,
    }


def _extract_first_author(author_part: str) -> str:
    candidate = (author_part or "").strip()
    if not candidate:
        return ""
    separators = [",", "，", "、", ";", "；", ".", "。", " and ", " AND ", " & "]
    first = candidate
    for sep in separators:
        pos = first.find(sep)
        if pos > 0:
            first = first[:pos]
    return first.strip()


def _extract_first_year(text: str) -> Optional[int]:
    match = YEAR_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalize_author_key(author: str) -> str:
    if not author:
        return ""
    value = str(author).strip()
    value = re.sub(r"\bet\s+al\.?\b", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+等$", "", value).strip()
    value = re.split(r"[,&，、;；]|\band\b", value, flags=re.IGNORECASE)[0].strip()
    if not value:
        return ""
    if _contains_cjk(value):
        return _normalize_cjk_sort_key(value)
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return folded.lower()


def _normalize_cjk_sort_key(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    if not compact:
        return ""
    try:
        return f"cn:{compact.encode('gb18030').hex()}"
    except UnicodeEncodeError:
        return f"cnu:{compact.encode('utf-8', errors='ignore').hex()}"


def _estimate_author_count(author_part: str) -> int:
    value = (author_part or "").strip()
    if not value:
        return 1
    if re.search(r"\bet\s+al\.\b", value, flags=re.IGNORECASE) or "等" in value:
        return 3
    tokens = [seg.strip() for seg in re.split(r"[，、;；]| & |\band\b", value, flags=re.IGNORECASE) if seg.strip()]
    if len(tokens) >= 2:
        return len(tokens)
    comma_split = [seg.strip() for seg in value.split(",") if seg.strip()]
    if len(comma_split) >= 2:
        return len(comma_split)
    return 1


def _estimate_western_author_count(author: str) -> int:
    cleaned = re.sub(r"\bet\s+al\.\b", "", author, flags=re.IGNORECASE)
    separators = re.split(r"\s*&\s*|\band\b|;", cleaned, flags=re.IGNORECASE)
    parts = [item.strip() for item in separators if item.strip()]
    if len(parts) >= 2:
        return len(parts)
    comma_parts = [item.strip() for item in cleaned.split(",") if item.strip()]
    if len(comma_parts) >= 2:
        return len(comma_parts)
    return 1


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _has_inline_page_locator(segment: str, year: int, suffix: str = "") -> bool:
    text = (segment or "").strip()
    year_text = str(year)
    suffix_text = (suffix or "").strip()

    # 页码应是年份后的第二层定位信息，避免把“作者, 2020”误判为含页码。
    patterns = [
        rf"{year_text}{suffix_text}\s*[:：]\s*\d+(?:\s*[-–]\s*\d+)?\s*$",
        rf"{year_text}{suffix_text}\s*[,，]\s*(?:pp?\.?\s*)?\d+(?:\s*[-–]\s*\d+)?\s*$",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _should_apply_cn_person_name_rules(author: str) -> bool:
    normalized = re.sub(r"\s+", "", str(author or ""))
    if not normalized:
        return False

    if re.search(r"[《》“”\"'()（）\[\]【】{}<>]", normalized):
        return False

    org_keywords = (
        "大学",
        "学院",
        "研究院",
        "研究所",
        "研究中心",
        "实验室",
        "委员会",
        "国务院",
        "政府",
        "部",
        "局",
        "厅",
        "中心",
        "集团",
        "公司",
        "出版社",
        "学会",
        "协会",
        "银行",
        "医院",
        "人民日报",
        "新闻网",
    )
    if any(keyword in normalized for keyword in org_keywords):
        return False

    # 超过常见人名长度且不含多作者分隔符，通常是机构/题名片段。
    if len(normalized) >= 6 and not re.search(r"[、和及,，;&]", normalized) and "等" not in normalized:
        return False

    return True


def _looks_multi_author(author: str) -> bool:
    value = (author or "").strip()
    if "等" in value:
        return True
    return bool(re.search(r"[、,，;&]|和|及", value))


def _normalize_chinese_author_display(author: str) -> str:
    text = (author or "").strip()
    if not text:
        return text
    compact = re.sub(r"\s+", "", text)
    parts = [
        seg for seg in re.split(
            r"(?:基于|根据|依据|针对|关于|对于|以及|比如|例如|，|,|。|；|;|：|:|\s)+",
            compact,
        )
        if seg
    ]
    tail = parts[-1] if parts else compact

    if tail.endswith("等"):
        name_part = tail[:-1]
        name_match = re.search(r"([\u4e00-\u9fff]{2,4})$", name_part)
        if name_match:
            return f"{name_match.group(1)}等"
        return tail

    if re.search(r"[、，,和及&]", tail):
        return tail

    single_tail = re.search(r"([\u4e00-\u9fff]{2,4})$", tail)
    if single_tail:
        return single_tail.group(1)
    return tail


def _extract_first_cn_author(author: str) -> str:
    text = (author or "").strip()
    if not text:
        return text
    text = re.sub(r"\s*等$", "", text)
    segments = [seg.strip() for seg in re.split(r"[、，,和及&]", text) if seg.strip()]
    return segments[0] if segments else text


def _build_cn_author_year_citation(author: str, year: Any, suffix: str = "") -> str:
    author_text = (author or "").strip()
    year_text = str(year or "").strip()
    suffix_text = (suffix or "").strip()
    if year_text:
        return f"{author_text}（{year_text}{suffix_text}）"
    return author_text


def _split_reference_core(reference_text: str) -> Tuple[str, str, str]:
    text = (reference_text or "").strip()
    if not text:
        return "", "", ""

    marker_match = re.search(r"[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]", text)
    prefix = text[: marker_match.start()].strip() if marker_match else text
    rest = text[marker_match.end():].strip() if marker_match else ""

    author_part = ""
    title_part = prefix
    if ". " in prefix:
        author_part, title_part = prefix.split(". ", 1)
    elif "。 " in prefix:
        author_part, title_part = prefix.split("。 ", 1)
    elif "." in prefix:
        author_part, title_part = prefix.split(".", 1)
    elif "。" in prefix:
        author_part, title_part = prefix.split("。", 1)

    author_part = author_part.strip(" .。")
    title_part = title_part.strip(" .。")
    return author_part, title_part, rest


def _explicit_author_count(author_part: str) -> int:
    value = (author_part or "").strip()
    if not value:
        return 0
    value = re.sub(r"[，,]?\s*等\s*$", "", value)
    value = re.sub(r",?\s*et\s+al\.?\s*$", "", value, flags=re.IGNORECASE)
    if not value:
        return 0
    chunks = [seg.strip() for seg in re.split(r"[,，、;；]| & |\band\b", value, flags=re.IGNORECASE) if seg.strip()]
    if chunks:
        return len(chunks)
    comma_split = [seg.strip() for seg in value.split(",") if seg.strip()]
    return len(comma_split) if comma_split else 1


def _is_org_abbreviation(author_part: str) -> bool:
    value = (author_part or "").strip()
    if not value or _contains_cjk(value):
        return False
    if "," in value:
        return False
    compact = re.sub(r"[\s\-.&/]", "", value)
    if len(compact) < 2 or len(compact) > 12:
        return False
    return compact.isupper()


def _looks_given_name_first(author_part: str) -> bool:
    value = (author_part or "").strip()
    if not value or _contains_cjk(value) or "," in value:
        return False
    if re.search(r"\bet\s+al\b", value, flags=re.IGNORECASE):
        return False

    tokens = [token for token in value.split() if token]
    if len(tokens) != 2:
        return False
    first, second = tokens
    if not (first[:1].isupper() and second[:1].isupper()):
        return False
    # 近似规则：若两个词均为完整英文单词，且未使用逗号，提示检查姓前名后
    return first.isalpha() and second.isalpha() and len(first) >= 3 and len(second) >= 3


def _is_foreign_title_all_caps(title: str) -> bool:
    letters = [ch for ch in title if ch.isalpha() and ch.isascii()]
    if len(letters) < 8:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper >= 6 and upper / max(len(letters), 1) > 0.75


def _extract_journal_name(reference_text: str) -> str:
    text = (reference_text or "").strip()
    match = re.search(
        r"[\[［]\s*J(?:/[A-Za-z]+)?\s*[\]］]\.\s*([^,，]+?)[,，]\s*(?:18|19|20)\d{2}",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _looks_abbreviated_journal_name(journal_name: str) -> bool:
    value = (journal_name or "").strip()
    if not value or _contains_cjk(value):
        return False
    if re.search(r"\b[A-Za-z]{1,6}\.", value):
        return True
    tokens = [token for token in re.split(r"\s+", value) if token]
    if len(tokens) >= 2 and all(len(token.strip(".,")) <= 4 for token in tokens):
        return True
    return False


def _has_journal_min_fields(reference_text: str) -> bool:
    text = (reference_text or "").strip()
    year_match = re.search(r"(?:18|19|20)\d{2}", text)
    if not year_match:
        return False

    has_volume_or_issue = bool(
        re.search(
            r"(?:18|19|20)\d{2}\s*(?:[,，]\s*)?(?:\d+\s*(?:\(\s*\d+\s*\))?|\(\s*\d+\s*\))",
            text,
        )
    )
    has_pages = bool(re.search(r"[:：]\s*[A-Za-z]?\d+(?:\s*[-–]\s*[A-Za-z]?\d+)?", text))
    return has_volume_or_issue and has_pages


def _has_non_arabic_numerals_for_biblio_fields(rest_part: str) -> bool:
    text = rest_part or ""
    checks = [
        r"第[一二三四五六七八九十百千〇零壹贰叁肆伍陆柒捌玖拾两]+版",
        r"[一二三四五六七八九十百千〇零壹贰叁肆伍陆柒捌玖拾两]+卷",
        r"[（(]\s*[一二三四五六七八九十百千〇零壹贰叁肆伍陆柒捌玖拾两]+\s*[）)]",
        r"[一二三四五六七八九十百千〇零壹贰叁肆伍陆柒捌玖拾两]+页",
    ]
    if any(re.search(pat, text) for pat in checks):
        return True
    if NON_ARABIC_NUM_RE.search(text) and re.search(r"(版|卷|期|页)", text):
        return True
    return False


def _language_rank(author: str) -> int:
    text = author or ""
    if re.search(r"[\u3040-\u30ff]", text):
        return 1  # 日文
    if re.search(r"[\u4e00-\u9fff]", text):
        return 0  # 中文
    if re.search(r"[\u0400-\u04ff]", text):
        return 3  # 俄文
    if re.search(r"[A-Za-z]", text):
        return 2  # 西文
    return 4      # 其他


def _sort_tuple(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item["lang_rank"],
        item["author_key"],
        item["single_rank"],
        item["year"],
        item["index"],
    )
