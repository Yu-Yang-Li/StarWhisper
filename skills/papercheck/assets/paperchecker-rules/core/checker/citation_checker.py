from .base_checker import BaseChecker
from models.document import Document
from models.compliance import ComplianceResult, CheckType
from typing import List, Dict, Any
import re
from core.checker.citation_checking.gbt7714_validator import (
    validate_reference_entry,
    validate_numeric_reference_numbering,
)
from core.checker.citation_checking.ucas_author_year_validator import (
    validate_ucas_author_year_body,
    validate_ucas_reference_notes,
    validate_ucas_reference_order,
)

class CitationChecker(BaseChecker):
    """引用合规性检查器 - 实现BaseChecker接口"""
    
    def get_check_type(self) -> CheckType:
        return CheckType.CITATIONS
    
    def get_check_name(self) -> str:
        return "citation_compliance_checker"
    
    def check(
        self,
        document: Document,
        author_format: str = "full",
        citation_standard: str = "legacy",
    ) -> ComplianceResult:
        """
        执行引用合规性检查
        
        Args:
            document: 要检查的文档对象
            
        Returns:
            ComplianceResult: 检查结果
        """
        results: List[Dict[str, Any]] = []  # 使用results名称以符合输出格式
        normalized_author_format = self._normalize_author_format(author_format)
        normalized_citation_standard = self._normalize_citation_standard(citation_standard)
        numeric_count = sum(1 for c in document.citations if c.format_type == "number")
        author_year_count = sum(1 for c in document.citations if c.format_type == "author_year")
        citation_style = self._infer_citation_style(numeric_count, author_year_count)

        statistics: Dict[str, Any] = {
            "total_citations": len(document.citations),
            "total_references": len(document.references),
            "matched_citations": 0,
            "unmatched_citations": 0,
            "year_inconsistencies": 0,
            "citation_numeric_count": numeric_count,
            "citation_author_year_count": author_year_count,
            "citation_style": citation_style,
            "citation_standard": normalized_citation_standard,
        }
        
        # 这里集成现有的引用匹配逻辑
        # 使用utils/reference_mapper中的函数
        from core.checker.citation_checking.reference_mapper import (
            classify_unmatched_author_year_citation,
            extract_authors_from_reference,
            format_citation_by_authors,
            map_author_year_citation_to_reference,
        )
        
        # 转换参考文献为字典格式以兼容映射函数
        reference_dicts = []
        for ref in document.references:
            reference_dicts.append({
                'text': ref.text,
                'author': ref.author,
                'year': ref.year if ref.year is not None else ref.text  # 使用参考文献文本作为备用来源
            })

        # 数字引用映射：编号 -> 参考文献
        reference_number_map: Dict[int, Dict[str, Any]] = {}
        for ref in reference_dicts:
            ref_no = self._extract_reference_no(ref.get('text', ''))
            if ref_no is not None and ref_no not in reference_number_map:
                reference_number_map[ref_no] = ref
        has_explicit_reference_indices = bool(reference_number_map)
        if numeric_count > 0 and not has_explicit_reference_indices:
            for idx, ref in enumerate(reference_dicts, 1):
                reference_number_map[idx] = ref

        matched_count = 0
        unmatched_count = 0
        corrected_count = 0
        formatted_count = 0
        corrections_needed = []
        formatting_needed = []
        reference_formatting_needed: List[Dict[str, Any]] = []
        reference_format_issues: List[Dict[str, Any]] = []
        citation_style_issues: List[Dict[str, Any]] = []
        unmatched_numeric_refs: List[str] = []
        unmatched_author_year_citations: List[str] = []
        unmatched_classification: List[Dict[str, Any]] = []
        unmatched_breakdown: Dict[str, int] = {
            "true_missing": 0,
            "ambiguous": 0,
            "extraction_noise": 0,
        }
        
        # 存储所有映射结果
        all_mapping_results = []
        
        # 处理每个引用
        for i, citation in enumerate(document.citations, 1):
            normalized_citation_text = citation.text
            if citation.format_type == "author_year":
                normalized_citation_text = self._normalize_author_year_citation_text(citation.text)

            report_entry = {
                "citation_index": i,
                "original_citation": normalized_citation_text,
                "matched": False,
                "needs_correction": False,
                "needs_formatting": False
            }
            
            if citation.format_type == 'author_year':
                # 使用现有的映射逻辑
                result = map_author_year_citation_to_reference(
                    normalized_citation_text,
                    reference_dicts,
                    strict_match=(normalized_citation_standard == "ucas"),
                )
                
                # 如果匹配成功，提取作者信息
                if result:
                    reference_text = result['reference']['text']
                    # 从映射结果中获取年份（优先使用映射结果中的年份）
                    ref_year = result['reference'].get('year', '')
                    
                    # 从参考文献文本中提取作者信息
                    authors, has_et_al = extract_authors_from_reference(reference_text)
                    
                    # 根据作者信息格式化引用
                    year = ref_year
                    if not year and 'corrected_citation' in result:
                        # 尝试从修正后的引用中提取年份
                        year_match = re.search(r'（(\d{4})）', result['corrected_citation'])
                        if year_match:
                            year = year_match.group(1)
                    
                    if year and authors:
                        formatted_citation = format_citation_by_authors(
                            authors,
                            year,
                            result['corrected_citation'],
                            has_et_al,
                            author_format=normalized_author_format,
                            citation_standard=normalized_citation_standard,
                        )
                        result['formatted_citation'] = formatted_citation
                    else:
                        result['formatted_citation'] = result['corrected_citation']
                        
                    # 记录映射结果
                    all_mapping_results.append(result)
                    
                    # 更新报告记录
                    matched_count += 1
                    report_entry.update({
                        "matched": True,
                        "reference_text": reference_text,
                        "reference_year": ref_year if ref_year else "未知",
                        "matched_year": ref_year if ref_year else "未知",
                        "authors": authors,
                        "has_et_al": has_et_al
                    })
                    
                    # 检查是否需要修正
                    needs_correction = result['corrected_citation'] != result['original_citation']
                    needs_formatting = result.get('formatted_citation', result['corrected_citation']) != result['corrected_citation']
                    
                    if needs_correction or needs_formatting:
                        if needs_correction:
                            corrected_count += 1
                            report_entry.update({
                                "corrected_citation": result['corrected_citation'],
                                "needs_correction": True,
                                "correction_reason": "年份不一致"
                            })
                            corrections_needed.append({
                                "original": result['original_citation'],
                                "corrected": result['corrected_citation'],
                                "reference": (reference_text[:100] + "...") if len(reference_text) > 100 else reference_text
                            })
                        
                        if needs_formatting:
                            formatted_count += 1
                            report_entry.update({
                                "formatted_citation": result['formatted_citation'],
                                "needs_formatting": True,
                                "formatting_reason": "作者格式不一致"
                            })
                            formatting_needed.append({
                                "original": result['corrected_citation'] if needs_correction else result['original_citation'],
                                "formatted": result['formatted_citation'],
                                "reference": (reference_text[:100] + "...") if len(reference_text) > 100 else reference_text
                            })
                        
                        # 如果已经有修正，则使用修正后的引用作为基准
                        if needs_correction:
                            report_entry["corrected_citation"] = result['corrected_citation']
                        else:
                            report_entry["corrected_citation"] = result['corrected_citation']
                            
                    # 如果只需要格式化而不需要修正
                    if needs_formatting and not needs_correction:
                        report_entry.update({
                            "needs_formatting": True,
                            "formatted_citation": result['formatted_citation'],
                            "formatting_reason": "作者格式不一致"
                        })
                        # 在这种情况下，仍然需要保留corrected_citation字段
                        report_entry["corrected_citation"] = result['corrected_citation']
                else:
                    unmatched_count += 1
                    unmatched_author_year_citations.append(normalized_citation_text)
                    classification = classify_unmatched_author_year_citation(
                        citation.text,
                        reference_dicts,
                        strict_match=(normalized_citation_standard == "ucas"),
                        normalized_citation=normalized_citation_text,
                    )
                    category = str(classification.get("category") or "true_missing")
                    if category not in unmatched_breakdown:
                        category = "true_missing"
                    unmatched_breakdown[category] += 1
                    classification.setdefault("label", category)
                    unmatched_classification.append(
                        {
                            "citation_index": i,
                            "citation_type": "author_year",
                            "original_citation": normalized_citation_text,
                            **classification,
                        }
                    )
            else:
                # 数字引用按编号映射到参考文献
                citation_no = self._extract_citation_no(citation.text)
                if citation_no is not None and citation_no in reference_number_map:
                    ref = reference_number_map[citation_no]
                    matched_count += 1
                    report_entry.update({
                        "matched": True,
                        "reference_text": ref["text"],
                        "reference_year": ref.get("year", "未知") if ref.get("year") else "未知",
                        "matched_year": ref.get("year", "未知") if ref.get("year") else "未知",
                    })
                    all_mapping_results.append({"reference": ref})
                else:
                    unmatched_count += 1
                    if citation_no is not None:
                        unmatched_numeric_refs.append(f"[{citation_no}]")
                        unmatched_breakdown["true_missing"] += 1
                        unmatched_classification.append(
                            {
                                "citation_index": i,
                                "citation_type": "number",
                                "original_citation": f"[{citation_no}]",
                                "category": "true_missing",
                                "label": "true_missing",
                                "reason_code": "NUMERIC_REFERENCE_MISSING",
                                "reason": "数字引文在文后参考文献中未找到对应编号。",
                            }
                        )
            
            results.append(report_entry)
        
        # 识别未被引用的参考文献
        used_references = {result['reference']['text'] for result in all_mapping_results if result}
        unused_references = [ref for ref in document.references if ref.text not in used_references]
        
        # 将unused_references转换为字典格式以兼容输出
        unused_references_dicts = []
        for ref in unused_references:
            unused_references_dicts.append({
                'text': ref.text,
                'author': ref.author,
                'year': ref.year
            })

        # 数字引用格式问题（例如 [32, 33]）
        numeric_spacing_issues = self._collect_numeric_spacing_issues(document, reference_number_map)
        formatted_count += len(numeric_spacing_issues)
        formatting_needed.extend(numeric_spacing_issues)

        # 引文体系混用（GB/T 7714：正文标注应统一采用一种体系）
        if citation_style == "mixed":
            style_issue = {
                "issue_code": "CITATION_STYLE_MIXED",
                "severity": "error",
                "message": "正文同时出现顺序编码制与著者-出版年制，建议统一为一种引文体系。",
                "suggestion": "全文统一采用顺序编码制或著者-出版年制，并保持文后参考文献组织方式一致。",
                "citation_numeric_count": numeric_count,
                "citation_author_year_count": author_year_count,
            }
            citation_style_issues.append(style_issue)
            formatted_count += 1
            formatting_needed.append({
                "original": "正文引文体系混用",
                "formatted": "建议统一为顺序编码制或著者-出版年制",
                "reference": f"数字制: {numeric_count} 条；著者-出版年制: {author_year_count} 条",
                "formatting_reason": style_issue["message"],
            })

        if unmatched_numeric_refs:
            preview = ", ".join(sorted(set(unmatched_numeric_refs), key=lambda x: int(x.strip("[]"))))
            missing_issue = {
                "issue_code": "CITATION_NUMERIC_REFERENCE_MISSING",
                "severity": "error",
                "message": "正文存在未在参考文献中找到对应条目的数字引文。",
                "suggestion": "补齐对应参考文献条目，或修正文中引文编号。",
                "missing_citations": preview,
            }
            citation_style_issues.append(missing_issue)

        if unmatched_author_year_citations and normalized_citation_standard == "ucas":
            unique_unmatched = list(dict.fromkeys(unmatched_author_year_citations))
            unmatched_by_category = self._group_unmatched_author_year_by_category(unmatched_classification)
            true_missing_items = unmatched_by_category.get("true_missing", [])
            preview_source = true_missing_items if true_missing_items else unique_unmatched
            preview = ", ".join(preview_source[:20])
            if len(preview_source) > 20:
                preview += f" ... (+{len(preview_source) - 20} more)"
            citation_style_issues.append({
                "issue_code": "CITATION_AUTHOR_YEAR_REFERENCE_MISSING",
                "severity": "error",
                "message": "存在未在参考文献中找到一一对应条目的著者-出版年制引文。",
                "suggestion": "补全文后参考文献，或修正文中作者/年份写法，确保一一对应。",
                "missing_citations": preview,
                "missing_citations_by_category": {
                    "true_missing": true_missing_items,
                    "ambiguous": unmatched_by_category.get("ambiguous", []),
                    "extraction_noise": unmatched_by_category.get("extraction_noise", []),
                },
            })
            if unmatched_by_category.get("ambiguous"):
                citation_style_issues.append({
                    "issue_code": "CITATION_AUTHOR_YEAR_REFERENCE_AMBIGUOUS",
                    "severity": "warn",
                    "message": "部分著者-出版年引文对应多个高相似候选条目，存在歧义。",
                    "suggestion": "补充作者全名/后缀或核对文后条目，消除一对多歧义。",
                    "ambiguous_citations": ", ".join(unmatched_by_category.get("ambiguous", [])[:20]),
                })
            if unmatched_by_category.get("extraction_noise"):
                citation_style_issues.append({
                    "issue_code": "CITATION_AUTHOR_YEAR_EXTRACTION_NOISE",
                    "severity": "warn",
                    "message": "部分未匹配引文疑似由抽取噪声导致。",
                    "suggestion": "优先核对原文排版/抽取质量，再判定是否真实缺失。",
                    "noise_citations": ", ".join(unmatched_by_category.get("extraction_noise", [])[:20]),
                })

        # 文后参考文献著录规则校验（GB/T 7714）
        for ref in document.references:
            reference_format_issues.extend(
                validate_reference_entry(ref.text, citation_style=citation_style)
            )

        # 数字制下，额外校验编号起始/连续性/重复
        if citation_style == "numeric":
            reference_format_issues.extend(
                validate_numeric_reference_numbering([ref.text for ref in document.references])
            )

        if normalized_citation_standard == "ucas":
            if citation_style in {"author_year", "mixed"}:
                ucas_body_issues = validate_ucas_author_year_body(document)
                citation_style_issues.extend(ucas_body_issues)
                ucas_body_formatting = self._convert_citation_issues_to_formatting_items(ucas_body_issues)
                formatting_needed.extend(ucas_body_formatting)
                formatted_count += len(ucas_body_formatting)

            if citation_style == "author_year":
                reference_format_issues.extend(validate_ucas_reference_order(document.references))

            reference_format_issues.extend(validate_ucas_reference_notes(document.references))

        if citation_style == "numeric" and not has_explicit_reference_indices:
            reference_format_issues = self._compact_numeric_missing_index_issues(
                reference_format_issues,
                len(document.references),
            )

        reference_formatting_items = self._convert_reference_issues_to_formatting_items(reference_format_issues)
        citation_formatting_needed = self._dedupe_formatting_items(formatting_needed)
        reference_formatting_needed = self._dedupe_formatting_items(reference_formatting_items)
        citation_formatted_count = len(citation_formatting_needed)
        reference_formatted_count = len(reference_formatting_needed)
        formatted_count = citation_formatted_count + reference_formatted_count
        issue_layer_summary = self._build_issue_layer_summary(reference_format_issues, citation_style_issues)
        
        # 更新统计信息
        statistics.update({
            "matched_citations": matched_count,
            "unmatched_citations": unmatched_count,
            "corrected_count": corrected_count,
            "formatted_count": formatted_count,
            "citation_formatted_count": citation_formatted_count,
            "reference_formatted_count": reference_formatted_count,
            "unused_references_count": len(unused_references),
            "reference_format_issue_count": len(reference_format_issues),
            "citation_style_issue_count": len(citation_style_issues),
            "unmatched_true_missing_count": unmatched_breakdown["true_missing"],
            "unmatched_ambiguous_count": unmatched_breakdown["ambiguous"],
            "unmatched_extraction_noise_count": unmatched_breakdown["extraction_noise"],
            "unmatched_breakdown": dict(unmatched_breakdown),
            "strong_rule_issue_count": issue_layer_summary["strong_rule_issue_count"],
            "heuristic_rule_issue_count": issue_layer_summary["heuristic_rule_issue_count"],
            "unclassified_rule_issue_count": issue_layer_summary["unclassified_rule_issue_count"],
            "high_confidence_issue_count": issue_layer_summary["high_confidence_issue_count"],
            "medium_confidence_issue_count": issue_layer_summary["medium_confidence_issue_count"],
            "low_confidence_issue_count": issue_layer_summary["low_confidence_issue_count"],
            "unclassified_confidence_issue_count": issue_layer_summary["unclassified_confidence_issue_count"],
        })
        
        # 创建兼容现有API格式的输出
        compatibility_output = {
            "test_date": __import__('datetime').datetime.now().isoformat(),
            "document": document.metadata.get('file_path', ''),
            "author_format": normalized_author_format,
            "citation_standard": normalized_citation_standard,
            "citation_style": citation_style,
            "citation_numeric_count": numeric_count,
            "citation_author_year_count": author_year_count,
            "total_citations": len(document.citations),
            "total_references": len(document.references),
            "results": results,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "corrected_count": corrected_count,
            "formatted_count": formatted_count,
            "citation_formatted_count": citation_formatted_count,
            "reference_formatted_count": reference_formatted_count,
            "unused_references_count": len(unused_references),
            "match_rate": f"{matched_count/len(document.citations)*100:.1f}%" if len(document.citations) > 0 else "0%",
            "corrections_needed": corrections_needed,
            # 保持向后兼容：formatting_needed 仅保留正文引用格式问题
            "formatting_needed": citation_formatting_needed,
            "citation_formatting_needed": citation_formatting_needed,
            "reference_formatting_needed": reference_formatting_needed,
            "unused_references": unused_references_dicts,
            "reference_format_issues": reference_format_issues,
            "citation_style_issues": citation_style_issues,
            "reference_format_issue_count": len(reference_format_issues),
            "citation_style_issue_count": len(citation_style_issues),
            "unmatched_true_missing_count": unmatched_breakdown["true_missing"],
            "unmatched_ambiguous_count": unmatched_breakdown["ambiguous"],
            "unmatched_extraction_noise_count": unmatched_breakdown["extraction_noise"],
            "unmatched_breakdown": dict(unmatched_breakdown),
            "unmatched_classification": unmatched_classification,
            "strong_rule_issue_count": issue_layer_summary["strong_rule_issue_count"],
            "heuristic_rule_issue_count": issue_layer_summary["heuristic_rule_issue_count"],
            "unclassified_rule_issue_count": issue_layer_summary["unclassified_rule_issue_count"],
            "high_confidence_issue_count": issue_layer_summary["high_confidence_issue_count"],
            "medium_confidence_issue_count": issue_layer_summary["medium_confidence_issue_count"],
            "low_confidence_issue_count": issue_layer_summary["low_confidence_issue_count"],
            "unclassified_confidence_issue_count": issue_layer_summary["unclassified_confidence_issue_count"],
            "rule_strength_counts": issue_layer_summary["rule_strength_counts"],
            "confidence_tier_counts": issue_layer_summary["confidence_tier_counts"],
        }
        
        # 创建结果对象，保留兼容输出作为metadata
        result = ComplianceResult(
            check_type=CheckType.CITATIONS,
            is_compliant=(
                unmatched_count == 0
                and len(reference_format_issues) == 0
                and len(citation_style_issues) == 0
            ),
            issues=results,  # 将结果以issues的形式存储
            statistics=statistics,
            metadata={
                "compatibility_output": compatibility_output,
                "checker_version": "1.3.0"
            }
        )
        
        return result

    def _extract_reference_no(self, reference_text: str):
        """从参考文献条目提取编号，如 [12] ..."""
        if not reference_text:
            return None
        match = re.match(r'^\s*[\[［〔【](\d+)[\]］〕】]', reference_text)
        if match:
            return int(match.group(1))
        return None

    def _extract_citation_no(self, citation_text: str):
        """从数字引用提取编号，如 [12]"""
        if not citation_text:
            return None
        match = re.match(r'^\s*[\[［〔【](\d+)[\]］〕】]\s*$', citation_text.strip())
        if match:
            return int(match.group(1))
        return None

    def _collect_numeric_spacing_issues(self, document: Document, reference_number_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        收集同一方括号内分隔符/空格不统一问题，例如：
        - [32, 33] -> [32,33]
        - [32，33] -> [32,33]
        仅在正文部分扫描，跳过目录行。
        """
        issues: List[Dict[str, Any]] = []
        seen = set()

        paragraphs = document.content or []
        body_paragraphs = paragraphs
        ref_heading_indices = [
            i for i, text in enumerate(paragraphs)
            if (text or "").strip().lower() in {"参考文献", "references"}
        ]
        if ref_heading_indices:
            body_paragraphs = paragraphs[:ref_heading_indices[-1]]

        # 捕获并列与区间组合，如 [32, 33] / [32，33] / [32;33] / [32-34, 36]
        token_pattern = re.compile(
            r'[\[［〔【](\d+(?:\s*[-–]\s*\d+)?(?:\s*[,，;；、]\s*\d+(?:\s*[-–]\s*\d+)?)*)[\]］〕】]'
        )
        for paragraph in body_paragraphs:
            if re.search(r'\t\s*\d+\s*$', (paragraph or "").strip()):
                continue
            for match in token_pattern.finditer(paragraph or ""):
                inner = match.group(1)
                original = f"[{inner}]"
                if original in seen:
                    continue
                seen.add(original)

                formatted_inner = re.sub(r'\s*[,，;；、]\s*', ',', inner)
                formatted_inner = re.sub(r'\s*[-–]\s*', '-', formatted_inner)
                formatted = f"[{formatted_inner}]"
                if formatted == original:
                    continue

                ref_preview = self._build_reference_preview_from_inner(inner, reference_number_map)
                issues.append({
                    "original": original,
                    "formatted": formatted,
                    "reference": ref_preview,
                    "formatting_reason": "同一方括号内编号分隔符或空格不统一，建议统一为英文逗号且不留空格。"
                })

        return issues

    def _build_reference_preview_from_inner(self, inner: str, reference_number_map: Dict[int, Dict[str, Any]]) -> str:
        """根据并列引用中的编号，拼接参考文献预览。"""
        numbers: List[int] = []
        for part in [p.strip() for p in re.split(r'[,，;；、]', inner)]:
            range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start <= end:
                    numbers.extend(range(start, end + 1))
                continue
            if part.isdigit():
                numbers.append(int(part))

        previews = []
        for no in numbers:
            ref = reference_number_map.get(no)
            if not ref:
                continue
            previews.append(f"[{no}] {ref.get('text', '')[:90]}")
            if len(previews) >= 3:
                break
        return " ; ".join(previews)

    def _convert_reference_issues_to_formatting_items(
        self,
        reference_issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将参考文献著录问题转换为前端兼容的 formatting_needed 条目。"""
        items: List[Dict[str, Any]] = []
        seen = set()
        for issue in reference_issues:
            reason = issue.get("message", "")
            original = issue.get("reference_text", "")
            suggestion = issue.get("suggestion", "")
            key = (original, reason, suggestion)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "original": original,
                "formatted": suggestion,
                "reference": issue.get("issue_code", ""),
                "formatting_reason": reason,
            })
        return items

    def _normalize_author_format(self, author_format: str) -> str:
        value = (author_format or "full").strip().lower()
        return value if value in {"full", "abbrev"} else "full"

    def _normalize_citation_standard(self, citation_standard: str) -> str:
        value = (citation_standard or "legacy").strip().lower()
        return value if value in {"legacy", "ucas"} else "legacy"

    def _infer_citation_style(self, numeric_count: int, author_year_count: int) -> str:
        if numeric_count == 0 and author_year_count == 0:
            return "unknown"
        if numeric_count > 0 and author_year_count == 0:
            return "numeric"
        if author_year_count > 0 and numeric_count == 0:
            return "author_year"
        return "mixed"

    def _convert_citation_issues_to_formatting_items(
        self,
        citation_issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen = set()
        for issue in citation_issues:
            reason = issue.get("message", "")
            original = issue.get("citation_normalized") or issue.get("citation_text", "")
            suggestion = issue.get("suggested_citation") or issue.get("suggestion", "")
            if original:
                original = self._normalize_author_year_citation_text(original)
            key = (original, reason, suggestion)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "original": original,
                "formatted": suggestion,
                "reference": issue.get("issue_code", ""),
                "formatting_reason": reason,
            })
        return items

    def _dedupe_formatting_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[tuple, Dict[str, Any]] = {}
        for item in items or []:
            original = str(item.get("original", "")).strip()
            formatted = str(item.get("formatted", "")).strip()
            key = (original, formatted)
            if key not in merged:
                merged[key] = dict(item)
                continue

            existing = merged[key]
            existing_ref = str(existing.get("reference", "")).strip()
            incoming_ref = str(item.get("reference", "")).strip()
            if self._is_rule_code_reference(existing_ref) and not self._is_rule_code_reference(incoming_ref):
                existing["reference"] = incoming_ref
            if not existing.get("formatting_reason") and item.get("formatting_reason"):
                existing["formatting_reason"] = item.get("formatting_reason")

        return list(merged.values())

    def _is_rule_code_reference(self, reference: str) -> bool:
        if not reference:
            return False
        return bool(re.fullmatch(r"[A-Z][A-Z0-9_]+", reference))

    def _normalize_author_year_citation_text(self, citation_text: str) -> str:
        text = (citation_text or "").strip()
        if not text:
            return text

        # 兼容 AI 提取包裹格式 [AUTH:...]
        if text.startswith("[AUTH:") and text.endswith("]"):
            text = text[6:-1].strip()

        text = re.sub(r'(?i)([A-Za-z]{2,})and([A-Z][A-Za-z]+)', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-z]{2,})etal\.?', r'\1 et al.', text)
        text = re.sub(r'\s*&\s*', ' & ', text)

        pair_pattern = re.compile(
            r"(?P<author>[^()（）]{1,80})\s*[（(](?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)\s*[）)]",
            re.IGNORECASE,
        )
        pair_matches = list(pair_pattern.finditer(text))
        if pair_matches:
            last = pair_matches[-1]
            author = (last.group("author") or "").strip()
            year = (last.group("year") or "").strip()
            suffix = (last.group("suffix") or "").strip()
            if re.search(r"[\u4e00-\u9fff]", author) and re.search(r"[A-Za-z]", author):
                cleaned_english = self._extract_tail_english_author(author)
                if cleaned_english:
                    return f"{cleaned_english} ({year}{suffix})"
            if re.search(r"[\u4e00-\u9fff]", author):
                cleaned_author = self._normalize_cn_author_fragment(author)
                return f"{cleaned_author}（{year}{suffix}）"
            return f"{author} ({year}{suffix})"

        english_pattern = re.compile(
            r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'\-\.]*(?:\s+(?:et al\.?|&|and|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'\-\.]*)){0,6})\s*[（(]((?:18|19|20)\d{2}[a-z]?)\s*[）)]",
            re.IGNORECASE,
        )
        english_matches = list(english_pattern.finditer(text))
        if english_matches:
            return english_matches[-1].group(0).strip()

        return text

    def _normalize_cn_author_fragment(self, author_text: str) -> str:
        compact = re.sub(r"\s+", "", str(author_text or ""))
        if not compact:
            return compact
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
            m = re.search(r"([\u4e00-\u9fff]{2,4})$", name_part)
            if m:
                return f"{m.group(1)}等"
            return tail

        if re.search(r"[、，,和及&]", tail):
            return tail

        m = re.search(r"([\u4e00-\u9fff]{2,4})$", tail)
        if m:
            return m.group(1)
        return tail

    def _extract_tail_english_author(self, author_text: str) -> str:
        text = re.sub(r'\s+', ' ', str(author_text or '')).strip(" ,，;；:：。.")
        if not text:
            return ""
        text = re.sub(r'(?i)([A-Za-z]{2,})and([A-Z][A-Za-z]+)', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-z]{2,})etal\.?', r'\1 et al.', text)
        text = re.sub(r'\s*&\s*', ' & ', text)
        match = re.search(
            r'([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+(?:\s+(?:&|and|et al\.?|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+)){0,8})\s*$',
            text,
            re.IGNORECASE,
        )
        if match:
            return re.sub(r'\s+', ' ', match.group(1)).strip()
        return ""

    def _build_issue_layer_summary(
        self,
        reference_issues: List[Dict[str, Any]],
        citation_issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        strength_counts: Dict[str, int] = {
            "strong_rule": 0,
            "heuristic_rule": 0,
            "unclassified": 0,
        }
        confidence_counts: Dict[str, int] = {
            "high": 0,
            "medium": 0,
            "low": 0,
            "unclassified": 0,
        }
        all_issues = list(reference_issues or []) + list(citation_issues or [])
        for issue in all_issues:
            strength = issue.get("rule_strength")
            if strength in {"strong_rule", "heuristic_rule"}:
                strength_counts[strength] += 1
            else:
                strength_counts["unclassified"] += 1

            tier = issue.get("confidence_tier")
            if tier in {"high", "medium", "low"}:
                confidence_counts[tier] += 1
            else:
                confidence_counts["unclassified"] += 1

        return {
            "strong_rule_issue_count": strength_counts["strong_rule"],
            "heuristic_rule_issue_count": strength_counts["heuristic_rule"],
            "unclassified_rule_issue_count": strength_counts["unclassified"],
            "high_confidence_issue_count": confidence_counts["high"],
            "medium_confidence_issue_count": confidence_counts["medium"],
            "low_confidence_issue_count": confidence_counts["low"],
            "unclassified_confidence_issue_count": confidence_counts["unclassified"],
            "rule_strength_counts": strength_counts,
            "confidence_tier_counts": confidence_counts,
        }

    def _group_unmatched_author_year_by_category(
        self,
        unmatched_classification: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {
            "true_missing": [],
            "ambiguous": [],
            "extraction_noise": [],
        }
        seen: Dict[str, set] = {
            "true_missing": set(),
            "ambiguous": set(),
            "extraction_noise": set(),
        }

        for item in unmatched_classification or []:
            if item.get("citation_type") != "author_year":
                continue
            category = str(item.get("category") or "true_missing")
            if category not in grouped:
                category = "true_missing"
            text = str(item.get("original_citation") or item.get("citation_text") or "").strip()
            if not text:
                continue
            if text in seen[category]:
                continue
            seen[category].add(text)
            grouped[category].append(text)

        return grouped

    def _compact_numeric_missing_index_issues(
        self,
        reference_issues: List[Dict[str, Any]],
        reference_count: int,
    ) -> List[Dict[str, Any]]:
        missing_index_issues = [
            issue for issue in (reference_issues or [])
            if issue.get("issue_code") == "REF_NUMERIC_MISSING_INDEX"
        ]
        if not missing_index_issues:
            return reference_issues or []

        retained = [
            issue for issue in (reference_issues or [])
            if issue.get("issue_code") != "REF_NUMERIC_MISSING_INDEX"
        ]
        retained.append({
            "issue_code": "REF_NUMERIC_MISSING_INDEX_BULK",
            "severity": "warn",
            "reference_text": f"共 {reference_count} 条参考文献未显式使用 [n] 编号。",
            "message": "顺序编码制正文已识别，但文后条目未显式编号，系统按文后顺序进行临时映射。",
            "suggestion": "建议为文后参考文献补充 [n] 顺序编号，以消除编号歧义。",
        })
        return retained
