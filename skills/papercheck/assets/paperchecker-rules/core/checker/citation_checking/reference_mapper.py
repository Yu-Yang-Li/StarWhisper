"""
引用映射工具
提供将引用与参考文献进行匹配和映射的功能
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def map_author_year_citation_to_reference(
    citation: str,
    references: List[Dict],
    strict_match: bool = False,
) -> Optional[Dict]:
    """
    将作者年份格式的引用映射到参考文献条目
    
    Args:
        citation (str): 作者年份格式的引用，如 "张三（2024）"
        references (list): 参考文献条目列表
        
    Returns:
        dict: 匹配的参考文献条目，如果没有匹配则返回None
    """
    # 从引用中提取作者和年份
    citation_author, citation_year = extract_author_year_from_citation(citation)
    
    if not citation_author or not citation_year:
        return None
    
    candidates = _collect_reference_match_candidates(citation_author, citation_year, references)
    if not candidates:
        return None
    best_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None

    base_threshold = 0.45 if strict_match else 0.3
    if best_candidate["total_score"] <= base_threshold:
        return None

    if strict_match and not _passes_strict_author_year_guard(
        citation_author=citation_author,
        citation_year=citation_year,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
    ):
        return None

    best_match = best_candidate["reference"]
    best_ref_year = best_candidate["ref_year"]
    # 如果找到了较好的匹配（得分阈值）
    if best_match:
        # 如果年份不一致，返回修正后的引用信息
        if best_ref_year and citation_year and best_ref_year != citation_year:
            # 创建一个包含修正年份的副本
            corrected_citation = f"{citation_author}（{best_ref_year}）"
            return {
                "reference": best_match,
                "corrected_citation": corrected_citation,
                "original_citation": citation
            }
        else:
            return {
                "reference": best_match,
                "corrected_citation": citation,  # 没有修正
                "original_citation": citation
            }
            
    return None


def classify_unmatched_author_year_citation(
    citation: str,
    references: List[Dict],
    strict_match: bool = False,
    normalized_citation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对未匹配的著者-出版年引文进行原因分类：
    - true_missing: 真缺失（正文与文后未一一对应）
    - ambiguous: 歧义（候选条目竞争，无法自动唯一映射）
    - extraction_noise: 抽取噪声（引文片段异常/不可解析）
    """
    raw_citation = (citation or "").strip()
    candidate_text = (normalized_citation or raw_citation).strip()
    citation_author, citation_year = extract_author_year_from_citation(candidate_text)

    if not citation_author or not citation_year:
        return _build_unmatched_classification(
            category="extraction_noise",
            reason_code="UNPARSEABLE_CITATION",
            reason="引文片段无法解析出标准作者-年份结构。",
            citation_text=candidate_text or raw_citation,
            raw_citation=raw_citation,
        )

    if _looks_like_extraction_noise_author(raw_citation, citation_author):
        return _build_unmatched_classification(
            category="extraction_noise",
            reason_code="NOISY_AUTHOR_FRAGMENT",
            reason="引文作者片段包含连接词/断裂痕迹，疑似抽取噪声。",
            citation_text=candidate_text,
            raw_citation=raw_citation,
            citation_author=citation_author,
            citation_year=citation_year,
        )

    candidates = _collect_reference_match_candidates(citation_author, citation_year, references)
    if not candidates:
        return _build_unmatched_classification(
            category="true_missing",
            reason_code="NO_REFERENCE_CANDIDATE",
            reason="参考文献中未找到可比较的候选条目。",
            citation_text=candidate_text,
            raw_citation=raw_citation,
            citation_author=citation_author,
            citation_year=citation_year,
        )

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    if _is_ambiguous_candidate_competition(citation_author, best, second):
        return _build_unmatched_classification(
            category="ambiguous",
            reason_code="AMBIGUOUS_CANDIDATES",
            reason="存在多个高相似候选条目，无法自动唯一映射。",
            citation_text=candidate_text,
            raw_citation=raw_citation,
            citation_author=citation_author,
            citation_year=citation_year,
            best_candidate=best,
            second_candidate=second,
        )

    base_threshold = 0.45 if strict_match else 0.3
    if float(best.get("total_score") or 0.0) <= base_threshold:
        category = "true_missing"
        reason_code = "BELOW_MATCH_THRESHOLD"
        reason = "最优候选相似度不足，未达到映射阈值。"
        if _looks_like_extraction_noise_author(raw_citation, citation_author) and float(best.get("author_similarity") or 0.0) < 0.5:
            category = "extraction_noise"
            reason_code = "LOW_SCORE_NOISY_FRAGMENT"
            reason = "引文片段疑似抽取噪声且与候选条目相似度偏低。"
        return _build_unmatched_classification(
            category=category,
            reason_code=reason_code,
            reason=reason,
            citation_text=candidate_text,
            raw_citation=raw_citation,
            citation_author=citation_author,
            citation_year=citation_year,
            best_candidate=best,
            second_candidate=second,
        )

    if strict_match and not _passes_strict_author_year_guard(
        citation_author=citation_author,
        citation_year=citation_year,
        best_candidate=best,
        second_candidate=second,
    ):
        if _is_ambiguous_candidate_competition(citation_author, best, second):
            return _build_unmatched_classification(
                category="ambiguous",
                reason_code="STRICT_GUARD_AMBIGUOUS",
                reason="严格模式下候选竞争导致拒配。",
                citation_text=candidate_text,
                raw_citation=raw_citation,
                citation_author=citation_author,
                citation_year=citation_year,
                best_candidate=best,
                second_candidate=second,
            )
        return _build_unmatched_classification(
            category="true_missing",
            reason_code="STRICT_GUARD_REJECTED",
            reason="严格模式校验未通过（作者/年份一致性不足）。",
            citation_text=candidate_text,
            raw_citation=raw_citation,
            citation_author=citation_author,
            citation_year=citation_year,
            best_candidate=best,
            second_candidate=second,
        )

    return _build_unmatched_classification(
        category="true_missing",
        reason_code="UNMATCHED_DEFAULT",
        reason="未匹配到唯一参考文献条目。",
        citation_text=candidate_text,
        raw_citation=raw_citation,
        citation_author=citation_author,
        citation_year=citation_year,
        best_candidate=best,
        second_candidate=second,
    )


def _collect_reference_match_candidates(
    citation_author: str,
    citation_year: str,
    references: List[Dict],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    citation_author_key = _normalize_author_key_for_strict(citation_author)
    _, citation_second_author = _extract_citation_author_parts(citation_author)
    citation_has_second_author = bool(citation_second_author)

    for ref in references:
        ref_text = str(ref.get("text", ""))
        ref_author, ref_year = extract_author_year_from_reference(ref_text)
        if not ref_author:
            continue

        author_score = calculate_author_match_score(citation_author, ref_text, ref_author)
        acronym_match = _is_acronym_author_match(citation_author, ref_author)
        if acronym_match:
            author_score = max(author_score, 0.9)

        second_author_hit = False
        if citation_second_author:
            second_author_hit = _reference_contains_author_token(ref_text, citation_second_author)
            if second_author_hit:
                author_score = max(author_score, 0.92)

        year_score = calculate_year_match_score(citation_year, ref_text, ref_year)
        total_score = author_score * 0.6 + year_score * 0.4

        ref_author_key = _normalize_author_key_for_strict(ref_author)
        exact_author_key_match = bool(citation_author_key and ref_author_key and citation_author_key == ref_author_key)
        if exact_author_key_match:
            author_score = max(author_score, 0.98)
        author_similarity = calculate_author_similarity(citation_author_key, ref_author_key)

        candidates.append(
            {
                "reference": ref,
                "ref_author": ref_author,
                "ref_year": ref_year,
                "author_score": author_score,
                "year_score": year_score,
                "total_score": total_score,
                "author_similarity": author_similarity,
                "citation_author_key": citation_author_key,
                "ref_author_key": ref_author_key,
                "exact_author_key_match": exact_author_key_match,
                "acronym_match": acronym_match,
                "second_author_hit": second_author_hit,
                "citation_has_second_author": citation_has_second_author,
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("total_score") or 0.0),
            bool(item.get("exact_author_key_match")),
            float(item.get("author_similarity") or 0.0),
        ),
        reverse=True,
    )
    return candidates


def _is_ambiguous_candidate_competition(
    citation_author: str,
    best_candidate: Optional[Dict[str, Any]],
    second_candidate: Optional[Dict[str, Any]],
) -> bool:
    if best_candidate is None or second_candidate is None:
        return False

    citation_key = _normalize_author_key_for_strict(citation_author)
    ref_key = str(best_candidate.get("ref_author_key") or "")
    is_chinese_pair = contains_chinese(citation_key) and contains_chinese(ref_key)
    min_author_similarity = 0.78 if is_chinese_pair else 0.72

    best_total = float(best_candidate.get("total_score") or 0.0)
    second_total = float(second_candidate.get("total_score") or 0.0)
    score_gap = best_total - second_total
    best_similarity = float(best_candidate.get("author_similarity") or 0.0)
    second_similarity = float(second_candidate.get("author_similarity") or 0.0)
    similarity_gap = best_similarity - second_similarity
    best_year_score = float(best_candidate.get("year_score") or 0.0)
    second_year_score = float(second_candidate.get("year_score") or 0.0)
    citation_has_second_author = bool(best_candidate.get("citation_has_second_author"))
    if best_candidate.get("exact_author_key_match") and not second_candidate.get("exact_author_key_match"):
        return False
    if citation_has_second_author:
        best_second_hit = bool(best_candidate.get("second_author_hit"))
        second_second_hit = bool(second_candidate.get("second_author_hit"))
        if best_second_hit != second_second_hit:
            return False

    year_competitive = second_year_score >= 1.0 or (best_year_score < 1.0 and second_year_score >= 0.9)
    if best_similarity >= 0.95 and similarity_gap >= 0.08:
        return False
    if score_gap < 0.08 and second_similarity >= min_author_similarity and year_competitive:
            return True
    return False


def _looks_like_extraction_noise_author(raw_citation: str, citation_author: str) -> bool:
    raw = (raw_citation or "").strip()
    author = _normalize_author_for_scoring(citation_author)
    if not author:
        return True

    if "<br" in raw.lower() or "\n" in raw:
        return True

    if re.match(r"^(?:据|由|与|及|并|和|在|对|关于|针对|根据|基于|依据|参考|采用|使用|比如|例如|如|以及|其中|对于|随着|通过|本文|文中|本研究)", author):
        return True

    compact_author = re.sub(r"\s+", "", author)
    if compact_author in {"学者", "等", "等学者", "研究者", "作者"}:
        return True
    if compact_author.startswith("等") and len(compact_author) <= 4:
        return True
    if re.search(r"\d", compact_author):
        return True
    if len(compact_author) <= 1:
        return True
    if not contains_chinese(compact_author) and re.match(r'^[a-z]', compact_author):
        return True
    if re.search(r"[<>{}\[\]|]", compact_author):
        return True
    if re.fullmatch(r"et\s*al\.?", compact_author, flags=re.IGNORECASE):
        return True
    if re.search(
        r"(?i)[A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,}etal\.?\s*[（(]",
        raw,
    ) and not re.search(r"(?i)\bet\s+al\.?\s*[（(]", raw):
        return True

    return False


def _build_unmatched_classification(
    *,
    category: str,
    reason_code: str,
    reason: str,
    citation_text: str,
    raw_citation: str,
    citation_author: Optional[str] = None,
    citation_year: Optional[str] = None,
    best_candidate: Optional[Dict[str, Any]] = None,
    second_candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "category": category,
        "reason_code": reason_code,
        "reason": reason,
        "citation_text": citation_text,
        "raw_citation": raw_citation,
    }
    if citation_author:
        payload["citation_author"] = citation_author
    if citation_year:
        payload["citation_year"] = citation_year

    if best_candidate is not None:
        ref = best_candidate.get("reference") or {}
        payload["best_candidate"] = {
            "reference_text": str(ref.get("text", "")),
            "ref_author": best_candidate.get("ref_author"),
            "ref_year": best_candidate.get("ref_year"),
            "total_score": round(float(best_candidate.get("total_score") or 0.0), 4),
            "author_similarity": round(float(best_candidate.get("author_similarity") or 0.0), 4),
            "author_score": round(float(best_candidate.get("author_score") or 0.0), 4),
            "year_score": round(float(best_candidate.get("year_score") or 0.0), 4),
        }
    if second_candidate is not None:
        ref = second_candidate.get("reference") or {}
        payload["second_candidate"] = {
            "reference_text": str(ref.get("text", "")),
            "ref_author": second_candidate.get("ref_author"),
            "ref_year": second_candidate.get("ref_year"),
            "total_score": round(float(second_candidate.get("total_score") or 0.0), 4),
            "author_similarity": round(float(second_candidate.get("author_similarity") or 0.0), 4),
            "author_score": round(float(second_candidate.get("author_score") or 0.0), 4),
            "year_score": round(float(second_candidate.get("year_score") or 0.0), 4),
        }
    return payload


def _passes_strict_author_year_guard(
    *,
    citation_author: str,
    citation_year: str,
    best_candidate: Dict[str, Any],
    second_candidate: Optional[Dict[str, Any]],
) -> bool:
    citation_key = _normalize_author_key_for_strict(citation_author)
    ref_key = best_candidate.get("ref_author_key", "")
    is_chinese_pair = contains_chinese(citation_key) and contains_chinese(ref_key)
    min_author_similarity = 0.78 if is_chinese_pair else 0.72

    best_author_similarity = float(best_candidate.get("author_similarity") or 0.0)
    best_author_score = float(best_candidate.get("author_score") or 0.0)
    best_year_score = float(best_candidate.get("year_score") or 0.0)
    best_ref_year = str(best_candidate.get("ref_year") or "")
    min_author_score = 0.85
    citation_acronym = _normalize_acronym(citation_author)
    if _looks_like_explicit_acronym(citation_author, citation_acronym):
        min_author_score = 0.8
    if best_candidate.get("acronym_match"):
        best_author_similarity = max(best_author_similarity, 0.9)
        best_author_score = max(best_author_score, 0.9)

    # 作者必须有实质性匹配，避免“仅按年份硬匹配”
    if best_author_similarity < min_author_similarity and best_author_score < min_author_score:
        return False

    # 若年份不一致，要求作者匹配更强，才允许把年份校正到参考文献年份
    if best_ref_year and citation_year and best_ref_year != citation_year and not is_ocr_error(citation_year, best_ref_year):
        if best_author_similarity < 0.90:
            return False

    # 若第一候选与第二候选分差过小，且第二候选也满足作者强匹配，则判定歧义，不自动匹配
    if second_candidate is not None:
        score_gap = float(best_candidate.get("total_score") or 0.0) - float(second_candidate.get("total_score") or 0.0)
        second_author_similarity = float(second_candidate.get("author_similarity") or 0.0)
        similarity_gap = best_author_similarity - second_author_similarity
        second_year_score = float(second_candidate.get("year_score") or 0.0)
        second_author_competes = True
        if best_candidate.get("exact_author_key_match") and not second_candidate.get("exact_author_key_match"):
            second_author_competes = False
        if best_candidate.get("citation_has_second_author"):
            second_author_competes = bool(second_candidate.get("second_author_hit"))
        year_competitive = second_year_score >= 1.0 or (best_year_score < 1.0 and second_year_score >= 0.9)
        if best_author_similarity >= 0.95 and similarity_gap >= 0.08:
            second_author_competes = False
        if score_gap < 0.08 and second_author_similarity >= min_author_similarity and second_author_competes and year_competitive:
            return False

    return True


def _normalize_author_key_for_strict(author: str) -> str:
    text = (author or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[A-Za-z])(?=[A-Z])", " ", text)
    text = re.sub(r"^\s*[\[［〔【]\d+[\]］〕】]\s*", "", text)
    text = re.sub(r"\bet\s+al\.?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*等\s*$", "", text)
    text = re.split(r"[，,、;；&]|(?:\band\b)|和|及", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip().lower()


def _normalize_author_for_scoring(author: str) -> str:
    text = str(author or "")
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r'\s+', ' ', text).strip(" ,，;；:：。.")
    text = re.sub(
        r'^(?:据|由|与|及|并|和|在|对|关于|针对|根据|基于|依据|参考|采用|使用|比如|例如|如|以及|其中|对于|随着|通过|本文|文中|本研究|该理论由|它起源于|该理论的核心观点由|尝试回应|借鉴|本研究借鉴)+',
        '',
        text,
    ).strip(" ,，;；:：。.")
    text = re.sub(r'([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])和([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])', r'\1 and \2', text)
    text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})\s*和\s*([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})', r'\1 and \2', text)
    text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})and([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+)', r'\1 and \2', text)
    text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})\s*等$', r'\1 et al.', text)
    text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})etal\.?', r'\1 et al.', text)
    text = re.sub(r'\s*&\s*', ' & ', text)
    if re.search(r'[\u4e00-\u9fff]', text) and re.search(r'[A-Za-z]', text):
        tail_match = re.search(
            r'([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+(?:\s+(?:&|and|et al\.?|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+)){0,8})\s*$',
            text,
            flags=re.IGNORECASE,
        )
        if tail_match:
            text = tail_match.group(1)
    return re.sub(r'\s+', ' ', text).strip(" ,，;；:：。.")


def _fold_latin_text(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")


def _tokenize_english_author(author: str) -> List[str]:
    normalized = _fold_latin_text(_normalize_author_for_scoring(author)).lower()
    tokens = [token for token in re.split(r'[^a-z0-9]+', normalized) if token]
    return [token for token in tokens if token not in {"et", "al"}]


def _normalize_acronym(text: str) -> str:
    return re.sub(r'[^A-Za-z]', '', _fold_latin_text(text)).upper()


def _looks_like_explicit_acronym(raw_author: str, normalized_acronym: str) -> bool:
    if not (2 <= len(normalized_acronym) <= 10 and normalized_acronym.isalpha()):
        return False
    compact = re.sub(r"[^A-Za-z]", "", str(raw_author or ""))
    if not compact:
        return False
    return compact.isupper()


def _build_author_acronym(author: str) -> str:
    normalized = _fold_latin_text(_normalize_author_for_scoring(author))
    words = [word for word in re.split(r'[^A-Za-z]+', normalized) if word]
    if len(words) < 2:
        return ""
    stop_words = {"of", "the", "and", "for", "in", "on", "to", "a", "an", "de", "la", "et", "al"}
    initials = [word[0].upper() for word in words if word.lower() not in stop_words]
    return "".join(initials)


def _is_acronym_author_match(author1: str, author2: str) -> bool:
    acronym1 = _normalize_acronym(author1)
    acronym2 = _normalize_acronym(author2)

    def _looks_like_acronym(value: str) -> bool:
        return 2 <= len(value) <= 10 and value.isalpha()

    if _looks_like_acronym(acronym1):
        return acronym1 == _build_author_acronym(author2)
    if _looks_like_acronym(acronym2):
        return acronym2 == _build_author_acronym(author1)
    return False


def _is_safe_author_containment_match(author1: str, author2: str) -> bool:
    normalized1 = _normalize_author_for_scoring(author1).lower()
    normalized2 = _normalize_author_for_scoring(author2).lower()
    if not normalized1 or not normalized2:
        return False

    if contains_chinese(normalized1) or contains_chinese(normalized2):
        return normalized1 in normalized2 or normalized2 in normalized1

    tokens1 = _tokenize_english_author(author1)
    tokens2 = _tokenize_english_author(author2)
    if not tokens1 or not tokens2:
        return False

    primary1 = tokens1[0]
    primary2 = tokens2[0]
    if primary1 == primary2:
        return True
    if len(primary1) >= 4 and primary1 in primary2:
        return True
    if len(primary2) >= 4 and primary2 in primary1:
        return True
    return False


def _normalize_citation_author_for_matching(author: str) -> str:
    normalized = _normalize_author_for_scoring(author)
    if not normalized:
        return ""
    if re.search(r'[A-Za-z]', normalized):
        normalized = re.sub(r'(?i)\bet\s+al\.?\b', 'et al.', normalized)
        normalized = re.sub(r'\s*&\s*', ' & ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def _extract_citation_author_parts(author: str) -> Tuple[str, str]:
    normalized = _normalize_author_for_scoring(author)
    if not normalized:
        return "", ""

    if contains_chinese(normalized):
        parts = [seg.strip() for seg in re.split(r"[、，,;；]|和|及|与|&", normalized) if seg.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return parts[0] if parts else "", ""

    cleaned = re.sub(r'(?i)\bet\s+al\.?\b', '', normalized).strip()
    parts = [seg.strip() for seg in re.split(r"\s*&\s*|\band\b|[,;]", cleaned, flags=re.IGNORECASE) if seg.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0] if parts else "", ""


def _reference_contains_author_token(reference_text: str, author_token: str) -> bool:
    token = _fold_latin_text(_normalize_author_for_scoring(author_token)).lower()
    text = _fold_latin_text(str(reference_text or "")).lower()
    if not token or not text:
        return False
    token_words = [word for word in re.split(r'[^a-z0-9]+', token) if word]
    if not token_words:
        return False
    primary = token_words[0]
    if len(primary) < 2:
        return False
    return bool(re.search(rf'\\b{re.escape(primary)}\\b', text))

def is_ocr_error(year1: str, year2: str) -> bool:
    """
    判断两个年份是否可能是OCR错误
    
    Args:
        year1 (str): 第一个年份
        year2 (str): 第二个年份
        
    Returns:
        bool: 是否可能是OCR错误
    """
    # 必须都是4位
    if len(year1) != 4 or len(year2) != 4:
        return False
    
    # 将OCR错误字符转换为标准数字
    ocr_to_digit = {
        'O': '0', 'o': '0',
        'I': '1', 'i': '1', 'l': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'G': '6', 'g': '6',
        'B': '8', 'b': '8',
        'D': '0', 'd': '0'
    }
    
    # 转换year1中的OCR错误字符
    normalized_year1 = ""
    for char in year1:
        if char in ocr_to_digit:
            normalized_year1 += ocr_to_digit[char]
        else:
            normalized_year1 += char
    
    # 转换year2中的OCR错误字符
    normalized_year2 = ""
    for char in year2:
        if char in ocr_to_digit:
            normalized_year2 += ocr_to_digit[char]
        else:
            normalized_year2 += char
    
    # 如果转换后相等，则是OCR错误
    if normalized_year1 == normalized_year2:
        return True
    
    # 计算不同的字符数
    diff_count = 0
    for i in range(4):
        if normalized_year1[i] != normalized_year2[i]:
            diff_count += 1
    
    # 如果只有一个字符不同，则可能是OCR错误
    if diff_count == 1:
        return True
    
    # 如果有两个字符不同，检查是否是常见的OCR错误对
    if diff_count == 2:
        # 常见的OCR错误对
        ocr_errors = [
            ('0', 'O'), ('1', 'I'), ('1', 'l'), ('2', 'Z'), 
            ('5', 'S'), ('6', 'G'), ('8', 'B'), ('0', 'D'),
            ('1', '7'), ('2', '7'), ('2', '1'), ('0', '8')
        ]
        
        # 检查不同的位置是否符合常见的OCR错误对
        diff_positions = []
        for i in range(4):
            if normalized_year1[i] != normalized_year2[i]:
                diff_positions.append(i)
        
        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions
            char1_1, char1_2 = normalized_year1[pos1], normalized_year1[pos2]
            char2_1, char2_2 = normalized_year2[pos1], normalized_year2[pos2]
            
            # 检查是否是OCR错误对
            for error1, error2 in ocr_errors:
                if (char1_1 == error1 and char2_1 == error2 and char1_2 == error2 and char2_2 == error1) or \
                   (char1_1 == error2 and char2_1 == error1 and char1_2 == error1 and char2_2 == error2):
                    return True
    
    return False


def calculate_author_match_score(citation_author: str, reference_text: str, ref_author: str) -> float:
    """
    计算作者在参考文献中的匹配得分
    
    Args:
        citation_author (str): 引用中的作者
        reference_text (str): 参考文献全文
        ref_author (str): 参考文献中提取的作者
        
    Returns:
        float: 匹配得分 (0.0 到 1.0)
    """
    citation_author = _normalize_author_for_scoring(citation_author)
    ref_author = _normalize_author_for_scoring(ref_author)
    reference_text = str(reference_text or "").lower()
    citation_author_lower = citation_author.lower()
    ref_author_lower = ref_author.lower()
    citation_author_fold = _fold_latin_text(citation_author_lower)
    ref_author_fold = _fold_latin_text(ref_author_lower)
    
    # 如果提取的作者完全匹配，给高分
    if citation_author_lower == ref_author_lower:
        return 1.0

    if _is_acronym_author_match(citation_author, ref_author):
        return 0.95

    # 仅在安全场景下允许包含匹配，避免 "An" 错匹配到 "Wang"
    if _is_safe_author_containment_match(citation_author, ref_author):
        return 0.9
    
    # 在整个参考文献文本中搜索作者名
    if citation_author_lower and (contains_chinese(citation_author_lower) or len(citation_author_lower) >= 4):
        lookup_key = citation_author_lower if contains_chinese(citation_author_lower) else citation_author_fold
        lookup_text = reference_text if contains_chinese(citation_author_lower) else _fold_latin_text(reference_text)
        if lookup_key and lookup_key in lookup_text:
            position = lookup_text.find(lookup_key)
            if position < 100:
                return 0.8
            if position < 200:
                return 0.6
            return 0.4

    citation_tokens = _tokenize_english_author(citation_author)
    ref_tokens = _tokenize_english_author(ref_author)
    if citation_tokens and ref_tokens:
        if citation_tokens[0] == ref_tokens[0]:
            return 0.9
        if len(citation_tokens[0]) >= 4 and citation_tokens[0] in ref_tokens[0]:
            return 0.75
    
    # 使用difflib计算相似度作为备选方案
    similarity = SequenceMatcher(None, citation_author_fold, ref_author_fold).ratio()
    return similarity * 0.5


def calculate_year_match_score(citation_year: str, reference_text: str, ref_year: str) -> float:
    """
    计算年份在参考文献中的匹配得分
    
    Args:
        citation_year (str): 引用中的年份
        reference_text (str): 参考文献全文
        ref_year (str): 参考文献中提取的年份
        
    Returns:
        float: 匹配得分 (0.0 到 1.0)
    """
    # 如果提取的年份完全匹配，给高分
    if ref_year and citation_year == ref_year:
        return 1.0
    
    # 检查是否为OCR错误
    if ref_year and is_ocr_error(citation_year, ref_year):
        return 0.9
    
    # 在整个参考文献文本中搜索年份
    if citation_year in reference_text:
        return 0.8
    
    return 0.0


def extract_author_year_from_citation(citation: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从作者年份格式的引用中提取作者和年份
    
    Args:
        citation (str): 作者年份格式的引用
        
    Returns:
        tuple: (作者, 年份)
    """
    stripped = (citation or "").strip()

    # 匹配格式：作者（年份），允许年份中包含OCR错误字符，支持后缀 a/b/c
    match = re.search(r'^(.+?)（([0-9OoIiZzSsGgBbDd]{4})([a-z]?)）', stripped, flags=re.IGNORECASE)
    if match:
        author = _normalize_citation_author_for_matching(match.group(1))
        year = match.group(2)
        suffix = (match.group(3) or "").lower()
        return (author, f"{year}{suffix}") if author else (None, None)
    
    # 匹配英文格式：Author (Year)，允许年份中包含OCR错误字符
    match = re.search(r'^(.+?)\s*\(([0-9OoIiZzSsGgBbDd]{4})([a-z]?)\)', stripped, flags=re.IGNORECASE)
    if match:
        author = _normalize_citation_author_for_matching(match.group(1))
        year = match.group(2)
        suffix = (match.group(3) or "").lower()
        return (author, f"{year}{suffix}") if author else (None, None)
        
    return None, None


def extract_author_year_from_reference(reference_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从参考文献条目中提取作者和年份
    
    Args:
        reference_text (str): 参考文献条目文本
        
    Returns:
        tuple: (作者, 年份)
    """
    # 匹配参考文献格式：作者. 文章标题. 期刊名, 年份, 卷(期): 页码.
    # 或者：作者. 文章标题[J]. 期刊名, 年份, 卷(期): 页码.
    # 或者：作者,作者. 文章标题. 期刊名, 年份, 卷(期): 页码.
    # 或者：作者. 书名[M]. 出版社: 年份.
    
    # 去除参考文献条目前的序号（如[1]、[71]等）
    clean_reference_text = re.sub(r'^\s*\[\d+\]\s*', '', reference_text)
    
    # 提取年份：仅接受 18xx/19xx/20xx，避免把页码(如 5476、1707)误识别为年份
    year = None
    year_token = r'((?:18|19|20)\d{2})'

    # 规则1：优先在文献类型标识 [J]/[M]/[R]/[D]/[C]/[N]/[S]/[Z] 之后找第一处年份
    # 这样可避开标题中的“2016-2035”等年份词，优先命中真实发表年份
    doc_type_match = re.search(r'[\[［][A-Za-z][\]］]', clean_reference_text)
    if doc_type_match:
        tail = clean_reference_text[doc_type_match.end():]
        tail_year_match = re.search(rf'(?<!\d){year_token}(?!\d)', tail)
        if tail_year_match:
            year = tail_year_match.group(1)

    # 规则2：常见位置模式兜底
    if not year:
        year_patterns = [
            rf'[\(（]{year_token}[\)）]',             # 括号中的年份
            rf'[，,]\s*{year_token}(?=[，,\.。\)])',  # 逗号后的年份
            rf'[:：]\s*{year_token}(?=[，,\.。])',    # 冒号后的年份（如出版地: 出版社, 2021）
            rf'(?<!\d){year_token}(?!\d)',           # 任意独立年份
        ]
        for pattern in year_patterns:
            year_match = re.search(pattern, clean_reference_text)
            if year_match:
                year = year_match.group(1)
                break
    
    # 提取作者（第一个逗号或点之前的内容）
    author_match = re.search(r'^([^.,]+)', clean_reference_text)
    author = author_match.group(1).strip() if author_match else None
    
    # 处理多个作者的情况，只取第一个作者
    if author and ('，' in author or ',' in author):
        # 分割多个作者，取第一个
        separators = ['，', ',']
        for sep in separators:
            if sep in author:
                author = author.split(sep)[0].strip()
                break
    
    # 如果作者以"等"或"et al"结尾，去掉这些词
    if author:
        author = re.sub(r'等$', '', author).strip()
        author = re.sub(r'et al\.?$', '', author).strip()
    
    return author, year


def calculate_author_similarity(author1: str, author2: str) -> float:
    """
    计算两个作者名的相似度

    Args:
        author1 (str): 第一个作者名
        author2 (str): 第二个作者名

    Returns:
        float: 相似度分数 (0.0 到 1.0)
    """
    # 检查输入是否为None
    if author1 is None:
        author1 = ""
    if author2 is None:
        author2 = ""

    # 统一归一化，避免上下文噪声/缩写差异造成误判
    author1 = _normalize_author_for_scoring(author1).lower()
    author2 = _normalize_author_for_scoring(author2).lower()
    author1_fold = _fold_latin_text(author1)
    author2_fold = _fold_latin_text(author2)
    
    # 完全匹配
    if author1 == author2:
        return 1.0
    if author1_fold and author2_fold and author1_fold == author2_fold:
        return 1.0
        
    # 使用difflib计算相似度
    if author1_fold and author2_fold:
        similarity = SequenceMatcher(None, author1_fold, author2_fold).ratio()
    else:
        similarity = SequenceMatcher(None, author1, author2).ratio()
    
    # 特殊处理：安全包含匹配（避免短词误匹配）
    if _is_safe_author_containment_match(author1, author2):
        similarity = max(similarity, 0.9)

    if _is_acronym_author_match(author1, author2):
        similarity = max(similarity, 0.95)
        
    # 特殊处理：英文名的姓氏匹配
    if is_english_name(author1) and is_english_name(author2):
        surname1 = extract_surname(author1)
        surname2 = extract_surname(author2)
        if surname1 == surname2:
            similarity = max(similarity, 0.95)
        elif surname1 in surname2 or surname2 in surname1:
            similarity = max(similarity, 0.9)
    
    # 特殊处理：多作者引用与单作者参考文献的匹配
    # 如果引用中有多个作者（包含&或and），尝试匹配第一个作者
    if ('&' in author1 or 'and' in author1) and not ('&' in author2 or 'and' in author2):
        # 分割引用中的第一个作者
        first_author_in_citation = author1.split('&')[0].split('and')[0].strip()
        # 计算第一个作者与参考文献作者的相似度
        first_author_similarity = SequenceMatcher(None, first_author_in_citation, author2).ratio()
        # 如果第一个作者相似度更高，则使用这个相似度
        if first_author_similarity > similarity:
            similarity = first_author_similarity
            # 特殊处理：如果第一个作者包含参考文献作者或反之
            if _is_safe_author_containment_match(first_author_in_citation, author2):
                similarity = max(similarity, 0.9)
            # 特殊处理：英文名的姓氏匹配
            if is_english_name(first_author_in_citation):
                first_surname = extract_surname(first_author_in_citation)
                second_surname = extract_surname(author2)
                if first_surname == second_surname:
                    similarity = max(similarity, 0.95)
                elif first_surname in second_surname or second_surname in first_surname:
                    similarity = max(similarity, 0.9)
    
    # 特殊处理：单作者引用与多作者参考文献的匹配
    # 如果参考文献中有多个作者（包含&或and），尝试匹配第一个作者
    if ('&' in author2 or 'and' in author2) and not ('&' in author1 or 'and' in author1):
        # 分割参考文献中的第一个作者
        first_author_in_reference = author2.split('&')[0].split('and')[0].strip()
        # 计算引用作者与第一个作者的相似度
        first_author_similarity = SequenceMatcher(None, author1, first_author_in_reference).ratio()
        # 如果第一个作者相似度更高，则使用这个相似度
        if first_author_similarity > similarity:
            similarity = first_author_similarity
            # 特殊处理：如果第一个作者包含引用作者或反之
            if _is_safe_author_containment_match(first_author_in_reference, author1):
                similarity = max(similarity, 0.9)
            # 特殊处理：英文名的姓氏匹配
            if is_english_name(first_author_in_reference):
                first_surname = extract_surname(author1)
                second_surname = extract_surname(first_author_in_reference)
                if first_surname == second_surname:
                    similarity = max(similarity, 0.95)
                elif first_surname in second_surname or second_surname in first_surname:
                    similarity = max(similarity, 0.9)
    
    # 特殊处理：处理参考文献中带有名字缩写的情况（如Ohanian, R.）
    # 如果参考文献中的作者包含逗号和名字缩写，而引用中只有姓氏
    if ',' in author2 and '.' in author2:
        # 提取参考文献中的姓氏部分（逗号前的部分）
        ref_surname = author2.split(',')[0].strip().lower()
        if ref_surname == author1:
            similarity = max(similarity, 0.95)
        elif ref_surname in author1 or author1 in ref_surname:
            similarity = max(similarity, 0.9)
    
    # 特殊处理：处理引用中只有姓氏，参考文献中有完整姓名的情况
    if ',' in author1 and '.' in author1:
        # 提取引用中的姓氏部分（逗号前的部分）
        cit_surname = author1.split(',')[0].strip().lower()
        if cit_surname == author2:
            similarity = max(similarity, 0.95)
        elif cit_surname in author2 or author2 in cit_surname:
            similarity = max(similarity, 0.9)
            
    return similarity


def is_similar_author(author1: str, author2: str) -> bool:
    """
    判断两个作者名是否相似 (兼容旧接口)
    
    Args:
        author1 (str): 第一个作者名
        author2 (str): 第二个作者名
        
    Returns:
        bool: 是否相似
    """
    return calculate_author_similarity(author1, author2) > 0.7


def is_english_name(name: str) -> bool:
    """判断是否为英文名"""
    # 检查输入是否为None
    if name is None:
        return False

    # 确保输入是字符串类型
    name = str(name)

    return bool(re.match(r'^[A-Za-z\s.,]+$', name))


def extract_surname(name: str) -> str:
    """提取英文名的姓氏"""
    import re

    # 检查输入是否为None
    if name is None:
        return ""

    # 确保输入是字符串类型
    name = str(name)

    # 去除"et al."后缀
    name = re.sub(r'\s+et\s+al\.?', '', name, flags=re.IGNORECASE).strip()

    # 去除"等"后缀
    name = re.sub(r'\s+等', '', name).strip()

    # 若为 "Surname, Given" 形式，逗号前通常是姓氏
    if ',' in name:
        return name.split(',', 1)[0].strip()

    # 分割名字部分
    parts = name.split()
    if parts:
        # 处理 "Surname Initial Initial" 形式，如 "Venuti M C" / "Ho R J Y"
        # 这类条目中首词为姓，后续均为名字首字母缩写
        def _is_initial_token(token: str) -> bool:
            token = token.strip()
            return bool(re.fullmatch(r'[A-Z]{1,4}\.?', token))

        if len(parts) >= 2 and all(_is_initial_token(token) for token in parts[1:]):
            return parts[0]

        # 特殊处理：对于像"Van Den Heuvel Christopher"这样的复合姓氏
        # 如果倒数第二个部分是常见的荷兰姓氏前缀，将它们组合起来
        dutch_prefixes = {'van', 'den', 'de', 'der', 'ter', 'ten', 'vanden', 'vander'}
        
        if len(parts) >= 2:
            # 检查倒数第二个部分是否为荷兰姓氏前缀
            second_last = parts[-2].lower()
            if second_last in dutch_prefixes:
                # 组合倒数两个部分作为姓氏
                return f"{parts[-2]} {parts[-1]}"
            # 检查倒数第三个和第二个部分是否都是荷兰姓氏前缀
            elif len(parts) >= 3 and parts[-3].lower() in dutch_prefixes and second_last in dutch_prefixes:
                # 组合倒数三个部分作为姓氏
                return f"{parts[-3]} {parts[-2]} {parts[-1]}"
        
        # 如果最后一个部分是单个字母或缩写，取倒数第二个部分作为姓氏
        if len(parts[-1]) == 1 or (len(parts[-1]) == 2 and parts[-1][1] == '.'):
            return parts[-2] if len(parts) > 1 else parts[-1]
        else:
            return parts[-1]  # 姓氏是最后一个部分
    return name


def extract_authors_from_reference(reference_text: str) -> Tuple[List[str], bool]:
    """
    从参考文献条目中提取作者列表

    参数:
        reference_text: 参考文献文本

    返回:
        作者列表, 是否有et al.标记
    """
    # 首先检查是否是中文文献
    if contains_chinese(reference_text[:50]) or any(char in reference_text for char in ['，', '。', '期刊', '杂志']):
        authors = extract_chinese_authors(reference_text)
        return authors, False
    else:
        return extract_english_authors(reference_text)


def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(re.search('[\u4e00-\u9fff]', text))


def extract_chinese_authors(reference_text: str) -> List[str]:
    """
    从中文参考文献条目中提取作者列表
    """
    authors = []

    # 中文文献模式: 作者1,作者2.文章名...
    chinese_pattern = r'^([^\.]+?)\.'  # 匹配第一个句号之前的内容

    match = re.search(chinese_pattern, reference_text)
    if match:
        authors_str = match.group(1)
        authors_str = re.sub(r'^\s*[\[［〔【]\d+[\]］〕】]\s*', '', authors_str)
        # 分割作者 (中文使用逗号或顿号分隔)
        if '，' in authors_str:
            authors = [author.strip() for author in authors_str.split('，') if author.strip()]
        elif ',' in authors_str:
            authors = [author.strip() for author in authors_str.split(',') if author.strip()]
        else:
            authors = [authors_str.strip()]

        # 处理"等"字
        authors = [author for author in authors if author != '等']

    return authors


def extract_english_authors(reference_text: str) -> List[str]:
    """
    从英文参考文献条目中提取作者列表

    返回:
        作者列表, 是否有et al.标记
    """
    authors = []
    has_et_al = False

    # 检查是否有et al.
    if re.search(r'et al\.?', reference_text, re.IGNORECASE):
        has_et_al = True

    normalized = re.sub(r'^\s*[\[［〔【]\d+[\]］〕】]\s*', '', reference_text or '').strip()

    # 提取作者部分（优先匹配“. 空格”前内容），避免在作者缩写点中截断。
    match = re.match(r'^(?P<authors>.+?)\.\s+', normalized)
    if match:
        authors_str = match.group('authors')
    else:
        first_dot = normalized.find('.')
        if first_dot == -1:
            return authors, has_et_al
        authors_str = normalized[:first_dot]

    # 处理"and"和"&"连接符
    authors_str = re.sub(r'\s+and\s+', ', ', authors_str)
    authors_str = re.sub(r'\s*&\s*', ', ', authors_str)

    # 移除"et al."等缩写
    authors_str = re.sub(r',?\s*et al\.?', '', authors_str, flags=re.IGNORECASE)

    # 改进的作者分割逻辑
    # 使用更智能的方法分割作者，考虑名字缩写中的逗号
    author_list = []
    current_author = ""
    in_abbreviation = False

    for char in authors_str:
        if char == ',' and not in_abbreviation:
            if current_author.strip():
                author_list.append(current_author.strip())
            current_author = ""
        else:
            current_author += char
            if char == '.':
                in_abbreviation = True
            elif char == ' ' and in_abbreviation:
                in_abbreviation = False

    if current_author.strip():
        author_list.append(current_author.strip())

    # 处理每个作者
    for author in author_list:
        # 处理名字缩写 (如 "A." 或 "A. B.")
        if re.match(r'^[A-Z]\.(?:\s*[A-Z]\.)?$', author):
            if authors:
                # 将缩写合并到前一个作者
                authors[-1] = authors[-1] + ' ' + author
            else:
                # 第一个就是缩写，可能是单字母作者名
                authors.append(author)
        else:
            authors.append(author)

    return authors, has_et_al


def format_citation_by_authors(
    authors: List[str],
    year: str,
    original_citation: str,
    has_et_al: bool = False,
    author_format: str = "full",
    citation_standard: str = "legacy",
) -> str:
    """
    根据作者数量和类型格式化引用

    参数:
        authors: 作者列表
        year: 年份
        original_citation: 原始引用
        has_et_al: 是否有et al.标记
        author_format: 双作者格式规则（full 或 abbrev）

    返回:
        格式化后的引用
    """
    if not authors:
        return original_citation

    normalized_author_format = (author_format or "full").strip().lower()
    if normalized_author_format not in {"full", "abbrev"}:
        normalized_author_format = "full"
    normalized_citation_standard = (citation_standard or "legacy").strip().lower()

    # 判断是否是中文作者
    is_chinese = contains_chinese(authors[0]) if authors else False

    if is_chinese:
        # 中文作者
        if len(authors) > 1 or has_et_al:
            if normalized_citation_standard == "ucas":
                # UCAS 著者-出版年制：中文多作者统一“第一作者 等（年份）”
                return f"{authors[0]} 等（{year}）"
            if len(authors) == 2 and not has_et_al and normalized_author_format == "full":
                return f"{authors[0]}和{authors[1]}（{year}）"
            return f"{authors[0]} 等（{year}）"
        else:
            return f"{authors[0]}（{year}）"
    else:
        # 欧美作者
        # 提取作者的姓
        surnames = [extract_surname(author) for author in authors]

        # 处理机构或团体作者
        if len(surnames) == 1 and (len(surnames[0].split()) > 2 or surnames[0].isupper()):
            return f"{surnames[0]}（{year}）"

        # abbrev 模式下，双作者也统一为 et al.
        if normalized_author_format == "abbrev" and len(surnames) >= 2:
            return f"{surnames[0]} et al.（{year}）"

        # 如果有et al.标记，或者作者数量大于2，使用et al.
        if has_et_al or len(surnames) > 2:
            return f"{surnames[0]} et al.（{year}）"
        elif len(surnames) == 2:
            return f"{surnames[0]} & {surnames[1]}（{year}）"
        else:
            return f"{surnames[0]}（{year}）"
