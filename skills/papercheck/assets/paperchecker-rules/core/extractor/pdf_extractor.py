import os
from .base_extractor import BaseExtractor
from models.document import Document, Citation, Reference
import re
from collections import Counter

class PDFExtractor(BaseExtractor):
    """PDF文档提取器"""
    
    def validate_file(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def extract(self, file_path: str) -> Document:
        """提取PDF文档内容"""
        # 尝试使用MinerU API进行转换
        md_content = None
        extraction_method = "mineru"
        extraction_warning = None
        try:
            from utils.mineru_pdf_converter import convert_pdf_to_markdown

            # 调用PDF转换功能
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, "config", "config.json")
            md_file_path = convert_pdf_to_markdown(file_path, config_path=config_path)

            # 从Markdown内容提取信息
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

        except Exception as e:
            extraction_warning = f"MinerU API转换失败，已切换到本地PDF提取: {e}"
            print(extraction_warning)
            # 使用本地方法提取PDF内容
            md_content = self._extract_pdf_locally(file_path)
            extraction_method = getattr(self, "_last_pdf_extraction_method", "local_pymupdf")

        # 提取正文内容（按行分割）
        paragraphs = md_content.split('\n') if md_content else []

        # 提取表格内容（从Markdown表格中）
        # 简单的表格识别
        tables_content = []
        lines = md_content.split('\n') if md_content else []
        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                tables_content.append(line.strip())

        # 提取引文和参考文献，这里需要实现PDF的提取逻辑
        citations = self._extract_citations(paragraphs, md_content or "", file_path)
        references = self._extract_references(paragraphs)

        # 构建文档对象
        return Document(
            content=paragraphs,
            tables=tables_content,
            citations=citations,
            references=references,
            metadata={
                'file_type': 'pdf',
                'file_path': file_path,
                'pdf_extraction_method': extraction_method,
                'pdf_extraction_warning': extraction_warning,
            }
        )

    def _extract_pdf_locally(self, file_path: str) -> str:
        """使用本地库提取PDF内容。

        优先用 pymupdf4llm 转 Markdown，保留标题、列表和表格等版式线索；
        若不可用或返回空文本，再回退到 PyMuPDF/fitz 的逐页文本抽取。
        """
        errors = []

        try:
            import pymupdf4llm

            markdown = pymupdf4llm.to_markdown(file_path)
            if isinstance(markdown, list):
                markdown = "\n\n".join(str(part) for part in markdown if part)
            markdown = self._normalize_local_pdf_text(str(markdown or ""))
            if self._has_usable_pdf_text(markdown):
                self._last_pdf_extraction_method = "pymupdf4llm"
                return markdown
            errors.append("pymupdf4llm returned empty or unusable text")
        except ImportError:
            errors.append("pymupdf4llm is not installed")
        except Exception as exc:
            errors.append(f"pymupdf4llm failed: {exc}")

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            errors.append("PyMuPDF/fitz is not installed")
            raise RuntimeError(self._local_pdf_error_message(errors)) from exc

        doc = None
        try:
            doc = fitz.open(file_path)
            page_texts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                try:
                    text = page.get_text("text", sort=True)
                except TypeError:
                    text = page.get_text("text")
                text = self._normalize_local_pdf_text(text)
                if text:
                    page_texts.append(f"\n\n<!-- page {page_num + 1} -->\n{text}")

            full_text = "\n".join(page_texts).strip()
            if not self._has_usable_pdf_text(full_text):
                errors.append("PyMuPDF/fitz extracted no usable text")
                raise RuntimeError(self._local_pdf_error_message(errors))
            self._last_pdf_extraction_method = "pymupdf_fitz"
            return full_text
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "Local PDF extraction failed" in str(exc):
                raise
            errors.append(f"PyMuPDF/fitz failed: {exc}")
            raise RuntimeError(self._local_pdf_error_message(errors)) from exc
        finally:
            if doc is not None:
                doc.close()

    def _normalize_local_pdf_text(self, text: str) -> str:
        """Normalize local PDF text without destroying paragraph boundaries."""
        if not text:
            return ""
        normalized_lines = []
        for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            normalized_lines.append(re.sub(r"[ \t]+", " ", line).strip())
        normalized = "\n".join(normalized_lines)
        normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
        return normalized.strip()

    def _has_usable_pdf_text(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return len(compact) >= 20

    def _local_pdf_error_message(self, errors: list) -> str:
        detail = "; ".join(errors)
        return (
            "Local PDF extraction failed. Install PyMuPDF and pymupdf4llm, or configure MINERU_API_KEY "
            "for layout-aware extraction. Details: "
            + detail
        )
    
    def _extract_citations(self, paragraphs: list, md_content: str, file_path: str) -> list:
        """提取引文"""
        all_citations = []
        processed_author_year_texts = set()  # 作者-年份引用去重

        references_start = self._find_reference_start(paragraphs)
        body_paragraphs = paragraphs
        if references_start != -1:
            body_paragraphs = paragraphs[:references_start]
        body_text = "\n".join(body_paragraphs)
        valid_reference_ids = self._collect_reference_indices(paragraphs, references_start)
        valid_reference_ids = valid_reference_ids or None
        
        # 从extractor模块导入AI增强的引用提取功能
        try:
            from .ai_extractor import extract_citations_from_text
            AI_EXTRACTION_AVAILABLE = True
        except ImportError:
            AI_EXTRACTION_AVAILABLE = False
        
        # 如果AI提取功能可用，使用AI从Markdown内容中提取作者年份格式的引用
        if AI_EXTRACTION_AVAILABLE:
            try:
                # 准备AI优化的配置（这里可以使用从配置文件传入的配置参数）
                ai_config = {
                    "api_key": "your-api-key",  # 可以从配置文件传入
                    "model_name": "qwen-plus"   # 可以从配置文件传入
                }
                
                # 从Markdown内容提取作者年份格式的引用
                text_citations = extract_citations_from_text(body_text, ai_config)
                
                # 将AI提取的引用添加到列表中
                for citation_text in text_citations:
                    normalized_text = self._normalize_extracted_author_year_text(citation_text)
                    if not normalized_text or normalized_text in processed_author_year_texts:
                        continue
                    if not self._is_ai_citation_supported_by_body(normalized_text, body_text):
                        continue
                    # 创建引用对象，格式化为AI提取的格式
                    citation_obj = Citation(
                        text=f"[AUTH:{normalized_text}]",
                        format_type='author_year',
                        context="AI提取"
                    )
                    all_citations.append(citation_obj)
                    processed_author_year_texts.add(normalized_text)
            except Exception as e:
                print(f"AI增强引用提取失败: {e}")

        for paragraph in body_paragraphs:
            if self._is_toc_entry(paragraph):
                continue
            for citation_id in self._extract_numeric_citation_ids(paragraph, valid_reference_ids):
                all_citations.append(Citation(text=f"[{citation_id}]", format_type='number', context=paragraph))
        
        # 提取作者年份格式引用作为补充（以防AI提取遗漏）
        for paragraph in body_paragraphs:
            author_year_citations = self._extract_author_year_citations(paragraph)
            for citation in author_year_citations:
                normalized_text = self._normalize_extracted_author_year_text(citation.text)
                if normalized_text and normalized_text not in processed_author_year_texts:
                    all_citations.append(citation)
                    processed_author_year_texts.add(normalized_text)
        
        return all_citations
    
    def _extract_author_year_citations(self, paragraph: str) -> list:
        """提取作者年份格式引用"""
        citations = []
        normalized_paragraph = re.sub(
            r'([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])和([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])',
            r'\1 and \2',
            paragraph or "",
        )
        normalized_paragraph = re.sub(
            r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})and([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+)',
            r'\1 and \2',
            normalized_paragraph,
        )
        chinese_pattern = re.compile(
            r'(?P<author>(?:[\u4e00-\u9fff]{2,4}(?:\s*[、和及]\s*[\u4e00-\u9fff]{2,4})*(?:\s*等)?))\s*[（(](?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)[）)]'
        )
        for match in chinese_pattern.finditer(normalized_paragraph):
            author = self._normalize_author_year_author(match.group("author"), is_chinese=True)
            if not author:
                continue
            year = match.group("year")
            suffix = (match.group("suffix") or "").lower()
            citations.append(Citation(
                text=f"{author}（{year}{suffix}）",
                format_type='author_year',
                author=author,
                year=f"{year}{suffix}",
                context=paragraph,
            ))

        english_pattern = re.compile(
            r'(?P<author>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+(?:\s+(?:&|and|et al\.?|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+)){0,6})\s*[（(](?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)[）)]',
            re.IGNORECASE,
        )
        for match in english_pattern.finditer(normalized_paragraph):
            author = self._normalize_author_year_author(match.group("author"), is_chinese=False)
            if not author:
                continue
            year = match.group("year")
            suffix = (match.group("suffix") or "").lower()
            citations.append(Citation(
                text=f"{author} ({year}{suffix})",
                format_type='author_year',
                author=author,
                year=f"{year}{suffix}",
                context=paragraph,
            ))
        
        return citations
    
    def _extract_references(self, paragraphs: list) -> list:
        """提取参考文献"""
        references = []

        # 方法0: 在参考文献区按 [n] 编号块合并（优先，解决跨行断裂）
        numbered_block_refs = self._extract_references_numbered_blocks(paragraphs)
        if len(numbered_block_refs) >= 30:
            # 当编号块提取已覆盖大量条目时，优先采用该结果，避免模式法引入噪声参考文献
            return self._remove_duplicates_and_validate(numbered_block_refs)

        references_start = self._find_reference_start(paragraphs)
        unnumbered_block_refs = self._extract_references_unnumbered_blocks(paragraphs, references_start)

        # 方法1: 查找"参考文献"标题后的传统方式
        traditional_refs = self._extract_references_traditional(paragraphs)

        # 无编号块提取覆盖充分时，优先使用块结果，避免传统逐行法引入碎片条目。
        if references_start != -1 and len(unnumbered_block_refs) >= 20:
            return self._remove_duplicates_and_validate(numbered_block_refs + unnumbered_block_refs)

        # 已在参考文献区提取到足够条目时，避免再跑高噪声全文兜底。
        if references_start != -1 and max(len(unnumbered_block_refs), len(traditional_refs)) >= 10:
            return self._remove_duplicates_and_validate(numbered_block_refs + unnumbered_block_refs + traditional_refs)

        # 兜底模式仅在参考文献区或文末窗口扫描，避免把正文句子误识别为参考文献。
        fallback_start_idx = references_start + 1 if references_start != -1 else int(len(paragraphs) * 0.7)

        # 方法2: 基于模式匹配的全文搜索方式
        pattern_based_refs = self._extract_references_pattern_based(paragraphs, start_idx=fallback_start_idx)

        # 方法3: 专门搜索文档后半部分的学术引用
        academic_refs = self._extract_references_academic_style(paragraphs, start_idx=fallback_start_idx)

        # 合并所有方法的结果
        all_refs = numbered_block_refs + unnumbered_block_refs + traditional_refs + pattern_based_refs + academic_refs

        # 去重并返回
        return self._remove_duplicates_and_validate(all_refs)

    def _extract_references_unnumbered_blocks(self, paragraphs: list, references_start: int = -1) -> list:
        """
        提取无编号参考文献：
        - 在参考文献区按“作者起始行”切块
        - 将跨行/跨页碎片合并成完整条目
        """
        references = []
        if not paragraphs:
            return references

        if references_start == -1:
            references_start = self._find_reference_start(paragraphs)
        if references_start == -1:
            config = self._load_config()
            start_percentage = config.get('academic_references_start_percentage', 0.7)
            references_start = int(len(paragraphs) * start_percentage)

        line_counts = Counter((p or "").strip() for p in paragraphs if (p or "").strip())
        current_lines = []

        for i in range(references_start + 1, len(paragraphs)):
            text = (paragraphs[i] or "").strip()
            if not text:
                continue
            if self._is_end_marker(text):
                break
            if self._is_reference_header_footer_line(paragraphs, i, text):
                continue
            if self._is_reference_noise_line(text, line_counts):
                continue

            if current_lines and self._looks_like_unnumbered_reference_start(text):
                if self._should_merge_with_current_reference(current_lines, text):
                    current_lines.append(text)
                    continue
                self._append_reference_from_lines(references, current_lines)
                current_lines = [text]
                continue

            current_lines.append(text)

        self._append_reference_from_lines(references, current_lines)
        return references

    def _is_reference_header_footer_line(self, paragraphs: list, index: int, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False

        has_year = bool(re.search(r'\b(19|20)\d{2}\b', stripped))
        has_doc_type = bool(re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]', stripped))
        if has_year or has_doc_type:
            return False
        if any(sep in stripped for sep in [',', '，', ':', '：', '.', '。']):
            return False
        if self._looks_like_unnumbered_reference_start(stripped):
            return False

        prev_text = ""
        for j in range(index - 1, -1, -1):
            prev_text = (paragraphs[j] or "").strip()
            if prev_text:
                break
        next_text = ""
        for j in range(index + 1, len(paragraphs)):
            next_text = (paragraphs[j] or "").strip()
            if next_text:
                break

        # 页眉/页脚常见形态：标题行与单独页码相邻
        if re.fullmatch(r'\d{1,4}', prev_text) or re.fullmatch(r'\d{1,4}', next_text):
            return len(stripped) >= 6

        return False

    def _looks_like_unnumbered_reference_start(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if re.match(r'^\s*\[\d+\]', stripped):
            return True
        # 中文作者起始：张三, 李四. / 张三，李四.
        if re.match(r'^(?:[\u4e00-\u9fff]{2,4}|[\u4e00-\u9fff]{2,20}(?:局|部|委|院|所|司|中心|集团|公司|大学|学院|学会|协会|研究院|政府|日报|新闻网|报|网|出版社))\s*[,，]\s*[\u4e00-\u9fffA-Za-z]', stripped):
            return True
        if re.match(r'^[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5}\.(?:\s+|$)', stripped):
            return True
        if re.match(r'^[A-Z][A-Za-z]+[A-Z]{2,}[A-Za-z]*\.(?:\s+|$)', stripped):
            return True
        if re.match(r"^[A-Z][A-Za-z'’&\-]+(?:\s+[A-Z][A-Za-z'’&\-]+){1,8}(?:\s+\([A-Z]{2,}\))?\.(?:\s+|$)", stripped):
            return True
        # 中文单作者/机构起始：方昊. / 国家能源局.
        if re.match(r'^[\u4e00-\u9fff]{2,4}\.', stripped):
            return True
        if re.match(r'^[\u4e00-\u9fff]{2,20}(?:局|部|委|院|所|司|中心|集团|公司|大学|学院|学会|协会|研究院|政府|日报|新闻网|报|网|出版社)\.', stripped):
            return True
        # 英文作者起始：Engle R F, Granger C W J. / Bağcı M, Soylu P K.
        if re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]+(?:\s+[A-Z]{1,3}(?:\.)?){0,4}\s*,\s*[A-ZÀ-ÖØ-Þ]", stripped):
            return True
        # 英文作者起始（无逗号）：Engle R F. / Johansen S. / Granger C W J.
        if re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]+(?:\s+[A-Z]{1,3}(?:\.)?){1,5}\.(?:\s+|$)", stripped):
            return True
        # PDF 换行导致作者姓单独成行：Bollerslev
        if re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]{2,}$", stripped):
            return True
        if re.match(r'^[A-Z][A-Z\s\.-]{2,30},\s*[A-Z]', stripped):
            return True
        return False

    def _is_reference_noise_line(self, text: str, line_counts: Counter) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        low = stripped.lower()
        if low in {'参考文献', '# 参考文献', 'references', '# references'}:
            return True
        if re.fullmatch(r'\d{1,4}', stripped):
            return True
        if stripped in {'致', '谢'}:
            return True
        if self._is_toc_entry(stripped):
            return True

        has_year = bool(re.search(r'\b(19|20)\d{2}\b', stripped))
        has_doc_type = bool(re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]', stripped))
        # 页眉/页脚类重复行：同一文本反复出现、无年份、无类型标识
        if line_counts.get(stripped, 0) >= 2 and not has_year and not has_doc_type and len(stripped) >= 8:
            return True

        return False

    def _extract_references_numbered_blocks(self, paragraphs: list) -> list:
        """
        在参考文献区按编号块提取条目：
        - 识别以 [n] 开头的行作为条目起点
        - 将后续非 [n] 行拼接到当前条目，直到下一个 [n] 或结束标记
        """
        references = []
        if not paragraphs:
            return references

        references_start = self._find_reference_start(paragraphs)
        if references_start == -1:
            config = self._load_config()
            start_percentage = config.get('academic_references_start_percentage', 0.7)
            references_start = int(len(paragraphs) * start_percentage)

        number_start_pattern = re.compile(r'^\s*\[(\d+)\]\s*(.*)$')
        current_lines = []

        for i in range(references_start + 1, len(paragraphs)):
            text = (paragraphs[i] or "").strip()
            if not text:
                continue
            if self._is_end_marker(text):
                break

            start_match = number_start_pattern.match(text)
            if start_match:
                self._append_reference_from_lines(references, current_lines)
                number = start_match.group(1)
                remainder = start_match.group(2).strip()
                if remainder:
                    current_lines = [f"[{number}] {remainder}"]
                else:
                    current_lines = [f"[{number}]"]
                continue

            if current_lines:
                # 参考文献区中的小标题通常很短且以#开头
                if text.startswith("#") and len(text) < 40:
                    continue
                current_lines.append(text)

        self._append_reference_from_lines(references, current_lines)
        return references

    def _append_reference_from_lines(self, references: list, lines: list):
        """将多行参考文献拼接为单条并写入结果。"""
        if not lines:
            return

        merged = " ".join(part.strip() for part in lines if part and part.strip())
        merged = re.sub(r'\s+', ' ', merged).strip()
        if not merged:
            return

        from core.checker.citation_checking.reference_mapper import extract_author_year_from_reference

        for candidate in self._split_compound_reference_entry(merged):
            extracted_author, extracted_year = extract_author_year_from_reference(candidate)
            references.append(Reference(
                text=candidate,
                author=extracted_author,
                year=extracted_year
            ))

    def _split_compound_reference_entry(self, merged: str) -> list:
        """拆分一行内被错误拼接的多条参考文献。"""
        if not merged:
            return []

        split_pattern = re.compile(
            r"(?<=[。\.])\s*(?=(?:"
            r"(?:[\u4e00-\u9fff]{2,4}|[\u4e00-\u9fff]{2,20}(?:局|部|委|院|所|司|中心|集团|公司|大学|学院|学会|协会|研究院|政府|日报|新闻网|报|网|出版社))\s*[,，]\s*[\u4e00-\u9fffA-Za-z]"
            r"|[\u4e00-\u9fff]{2,4}\."
            r"|[\u4e00-\u9fff]{2,20}(?:局|部|委|院|所|司|中心|集团|公司|大学|学院|学会|协会|研究院|政府|日报|新闻网|报|网|出版社)\."
            r"|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5}\.(?:\s+|$)"
            r"|[A-Z][A-Za-z]+[A-Z]{2,}[A-Za-z]*\.(?:\s+|$)"
            r"|[A-Z][A-Za-z'’&\-]+(?:\s+[A-Z][A-Za-z'’&\-]+){1,8}(?:\s+\([A-Z]{2,}\))?\.(?:\s+|$)"
            r"|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]+(?:\s+[A-Z]{1,3}(?:\.)?){0,4}\s*,\s*[A-ZÀ-ÖØ-Þ]"
            r"|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]+(?:\s+[A-Z]{1,3}(?:\.)?){1,5}\.(?:\s+|$)"
            r"|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]+\s+[A-Z]{1,3}\.(?=[A-Z])"
            r"))"
        )

        parts = [part.strip() for part in split_pattern.split(merged) if part and part.strip()]
        if not parts:
            return [merged]
        return self._merge_split_reference_fragments(parts)

    def _extract_references_traditional(self, paragraphs: list) -> list:
        """传统的参考文献提取方法（保留向后兼容）"""
        references = []
        references_start = self._find_reference_start(paragraphs)

        if references_start != -1:
            # 提取参考文献条目（从标题之后开始）
            for i in range(references_start + 1, len(paragraphs)):
                text = paragraphs[i].strip()
                if text:
                    # 检查是否为参考文献条目
                    if self._is_reference_entry(text):
                        # 提取DOI、URL等信息
                        doi = self._extract_doi(text)
                        url = self._extract_url(text)

                        # 从参考文献文本中提取作者和年份
                        from core.checker.citation_checking.reference_mapper import extract_author_year_from_reference
                        extracted_author, extracted_year = extract_author_year_from_reference(text)

                        references.append(Reference(
                            text=text,
                            author=extracted_author,
                            year=extracted_year
                        ))
                    # 如果遇到结束标记则停止
                    elif self._is_end_marker(text):
                        break
        else:
            # 如果没有找到"参考文献"标题，尝试在文档的后半部分查找可能的参考文献条目
            start_search = max(0, len(paragraphs) - 100)
            for i in range(start_search, len(paragraphs)):
                text = paragraphs[i].strip()
                if text and self._is_reference_entry(text):
                    doi = self._extract_doi(text)
                    url = self._extract_url(text)

                    references.append(Reference(
                        text=text,
                        author=None,  # 从参考文献文本中提取作者和年份
                        year=None
                    ))

        return references

    def _extract_references_pattern_based(self, paragraphs: list, start_idx: int = 0) -> list:
        """基于正则表达式模式的参考文献提取"""
        references = []

        # 定义多种参考文献模式
        patterns = [
            # [J] 期刊文章模式: 作者, 年份, 期刊名, [J], 卷(期): 页码.
            r'.*?[，,].*?\d{4}.*?\[J\].*?',
            # [M] 书籍模式: 作者. 书名[M]. 出版社, 年份.
            r'.*?[，,].*?\d{4}.*?\[M\].*?',
            # [C] 会议论文模式
            r'.*?[，,].*?\d{4}.*?\[C\].*?',
            # [D] 学位论文模式
            r'.*?[，,].*?\d{4}.*?\[D\].*?',
            # 简单年份模式（作者，年份，期刊）
            r'.*?[，,].*?\d{4}.*?[，,].*?[。\.]',
            # 序号模式 [数字]或(数字)
            r'[\[\(]\d+[\]\)].*?\d{4}.*?[。\.]',
            # 作者等年份模式：作者等，年份
            r'.*?等[，,]?\s*\d{4}.*?[。\.]',
            # 英文作者年份模式：Author (Year)
            r'[A-Z][a-z]+.*?\(\d{4}\)',
            # 英文作者年份模式：Author [Year]
            r'[A-Z][a-z]+.*?\[\d{4}\]',
            # 英文作者年份模式：Author, Year
            r'[A-Z][a-z]+.*?[,，]\s*\d{4}.*?[。\.]',
        ]

        start_idx = max(0, min(start_idx, len(paragraphs)))
        for i, paragraph in enumerate(paragraphs[start_idx:], start=start_idx):
            text = paragraph.strip()
            if len(text) > 20:  # 基本长度要求
                # 检查是否包含年份（基本要求）
                has_year = bool(re.search(r'\d{4}', text))
                if has_year:
                    # 检查是否符合任一模式
                    matches_pattern = any(re.search(pattern, text) for pattern in patterns)

                    # 额外检查：包含学术特征词汇
                    has_academic_indicators = any(indicator in text for indicator in
                                                 ['学报', '期刊', '研究', '出版', '出版社', '大学',
                                                  '论文', '文献', '杂志', '科学', '经济', '管理',
                                                  'Journal', 'Research', 'Studies', 'Review',
                                                  'University', 'Press', 'Academic'])

                    # 检查是否有引用格式特征
                    has_citation_indicators = any(indicator in text for indicator in
                                                 ['[J]', '[M]', '[C]', '[D]', '[S]', '[R]',
                                                  'Vol.', 'No.', 'pp.', 'p.', 'Vol',
                                                  '等.', '著', '编', '译'])

                    if matches_pattern or has_academic_indicators or has_citation_indicators:
                        # 验证是否为有效的参考文献（避免误判）
                        if self._is_valid_reference(text):
                            from core.checker.citation_checking.reference_mapper import extract_author_year_from_reference
                            extracted_author, extracted_year = extract_author_year_from_reference(text)

                            references.append(Reference(
                                text=text,
                                author=extracted_author,
                                year=extracted_year
                            ))

        return references

    def _extract_references_academic_style(self, paragraphs: list, start_idx: int = None) -> list:
        """专门针对学术论文风格的参考文献提取"""
        references = []

        # 从配置文件加载参数
        config = self._load_config()
        start_percentage = config.get('academic_references_start_percentage', 0.7)

        # 重点关注文档后半部分（通常参考文献在此）
        # 使用文档长度的百分比来确定搜索起始位置
        doc_length = len(paragraphs)

        if doc_length == 0:
            return references

        # 计算基于百分比的起始位置
        computed_start_idx = int(doc_length * start_percentage)
        if start_idx is None:
            start_idx = computed_start_idx
        else:
            start_idx = max(0, min(start_idx, doc_length))

        for i in range(start_idx, len(paragraphs)):
            text = paragraphs[i].strip()
            if len(text) > 10 and text:  # 有效文本
                # 检查是否符合学术参考文献的特征
                if self._has_academic_reference_characteristics(text):
                    from core.checker.citation_checking.reference_mapper import extract_author_year_from_reference
                    extracted_author, extracted_year = extract_author_year_from_reference(text)

                    references.append(Reference(
                        text=text,
                        author=extracted_author,
                        year=extracted_year
                    ))

        return references

    def _load_config(self) -> dict:
        """加载配置文件"""
        import json

        config_path = "config.json"
        if not os.path.exists(config_path):
            # 如果当前目录没有配置文件，尝试在项目根目录查找
            import pathlib
            project_root = pathlib.Path(__file__).parent.parent
            config_path = project_root / "config.json"

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            # 返回PDF提取器配置部分
            return full_config.get('pdf_extractor_config', {})
        else:
            # 如果配置文件不存在，返回默认值
            return {
                'academic_references_start_percentage': 0.7
            }

    def _has_academic_reference_characteristics(self, text: str) -> bool:
        """检查文本是否具有学术参考文献的特征"""
        # 包含年份
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', text))
        if not has_year:
            return False

        # 仅使用强信号：文献类型标识、卷期页、DOI/URL、编号开头等
        academic_indicators = [
            '[J]', '[M]', '[C]', '[D]', '[S]', '[R]',  # 文献类型标识
            '[N]', '[P]', '[EB/OL]',
            '学报', '期刊', '出版社', '大学', '杂志',
            'Vol.', 'No.', 'pp.', 'p.',  # 英文学术标识
            'ISBN', 'ISSN', 'DOI', 'doi', 'http://', 'https://',
            'University', 'Press', 'Journal',
            # 序号格式
            r'^\s*\d+\.', r'^\s*\[\d+\]', r'^\s*[A-Z]\d+\s+'
        ]

        for indicator in academic_indicators[:-2]:  # 前面的直接字符串检查
            if indicator in text:
                return True

        # 检查序号格式
        for pattern in academic_indicators[-2:]:  # 最后两个是正则表达式
            if re.search(pattern, text):
                return True

        # 卷期页或年份后定位符等结构化信息
        if re.search(r'(?:\b\d+\(\d+\)\s*[:：]\s*[A-Za-z]?\d+|[:：]\s*\d+\s*[-–]\s*\d+|doi\s*[:：]|https?://)', text, re.IGNORECASE):
            return True

        return False

    def _collect_reference_indices(self, paragraphs: list, references_start: int = -1) -> set:
        if not paragraphs:
            return set()
        if references_start == -1:
            references_start = self._find_reference_start(paragraphs)
        if references_start == -1:
            return set()

        number_start_pattern = re.compile(r'^\s*[\[［〔【](\d+)[\]］〕】]')
        indices = set()
        for i in range(references_start + 1, len(paragraphs)):
            text = (paragraphs[i] or "").strip()
            if not text:
                continue
            if self._is_end_marker(text):
                break
            match = number_start_pattern.match(text)
            if not match:
                continue
            number = int(match.group(1))
            if number > 0:
                indices.add(number)
        return indices

    def _extract_numeric_citation_ids(self, text: str, valid_reference_ids=None) -> list:
        if self._is_noise_like_numeric_context(text):
            return []

        ids = []
        token_pattern = re.compile(
            r'[\[［〔【](\d+(?:\s*[-–]\s*\d+)?(?:\s*[,，;；、]\s*\d+(?:\s*[-–]\s*\d+)?)*)[\]］〕】]'
        )
        for match in token_pattern.finditer(text or ""):
            raw = match.group(1)
            parts = [part.strip() for part in re.split(r'[,，;；、]', raw)]
            for part in parts:
                range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if start <= end:
                        for idx in range(start, end + 1):
                            if idx <= 0:
                                continue
                            if valid_reference_ids is not None and idx not in valid_reference_ids:
                                continue
                            ids.append(idx)
                    continue

                if part.isdigit():
                    idx = int(part)
                    if idx <= 0:
                        continue
                    if valid_reference_ids is not None and idx not in valid_reference_ids:
                        continue
                    ids.append(idx)

        return ids

    def _is_noise_like_numeric_context(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True

        if re.fullmatch(
            r'[\[［〔【]\d+(?:\s*[-–]\s*\d+)?(?:\s*[,，;；、]\s*\d+(?:\s*[-–]\s*\d+)?)*[\]］〕】]',
            stripped,
        ) and len(stripped) <= 16:
            return True

        math_operator_count = len(re.findall(r'[=+\-*/<>≤≥∑Σ∫]', stripped))
        alpha_count = len(re.findall(r'[A-Za-z\u4e00-\u9fff]', stripped))
        if math_operator_count >= 2 and alpha_count <= 3:
            return True

        return False

    def _normalize_author_year_author(self, author_text: str, is_chinese: bool) -> str:
        text = re.sub(r'<br\s*/?>', ' ', str(author_text or ''), flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip(" ,，;；:：。.")
        if not text:
            return ""

        leading_connectors = (
            r"(?:据|由|与|及|并|和|在|对|关于|针对|根据|基于|依据|参考|采用|使用|比如|例如|如|以及|其中|对于|随着|通过|本文|文中|本研究)"
        )
        text = re.sub(rf"^(?:{leading_connectors})+", "", text).strip(" ,，;；:：。.")
        text = re.sub(r'^(?:本研究借鉴|借鉴|尝试回应|该理论由|它起源于|该理论的核心观点由)', '', text).strip(" ,，;；:：。.")
        text = re.sub(r'([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])和([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})and([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+)', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})etal\.?', r'\1 et al.', text)
        text = re.sub(r'\s*&\s*', ' & ', text)

        if is_chinese:
            match = re.search(r'([\u4e00-\u9fff]{2,4}(?:\s*[、和及]\s*[\u4e00-\u9fff]{2,4})*(?:\s*等)?)$', text)
            if match:
                text = match.group(1)
        else:
            match = re.search(
                r'([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+(?:\s+(?:&|and|et al\.?|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ\'’\-]+)){0,6})$',
                text,
                flags=re.IGNORECASE,
            )
            if match:
                text = match.group(1)
            text = re.sub(r'\betal\b', 'et al.', text, flags=re.IGNORECASE)

        return re.sub(r'\s+', ' ', text).strip()

    def _normalize_extracted_author_year_text(self, citation_text: str) -> str:
        text = (citation_text or "").strip()
        if text.startswith("[AUTH:") and text.endswith("]"):
            text = text[6:-1].strip()
        text = re.sub(r'<br\\s*/?>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("（", "(").replace("）", ")")
        text = re.sub(r'([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])和([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ])', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})and([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+)', r'\1 and \2', text)
        text = re.sub(r'(?i)([A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]{2,})etal\.?', r'\1 et al.', text)
        text = re.sub(r'\s*&\s*', ' & ', text)

        pair_pattern = re.compile(
            r'(?P<author>[^()]{1,120}?)\s*[\\(](?P<year>(?:18|19|20)\d{2})(?P<suffix>[a-z]?)[\\)]',
            re.IGNORECASE,
        )
        matches = list(pair_pattern.finditer(text))
        if not matches:
            return ""
        match = matches[-1]
        author = (match.group("author") or "").strip()
        year = (match.group("year") or "").strip()
        suffix = (match.group("suffix") or "").lower()
        if not author or not year:
            return ""

        has_english = bool(re.search(r'[A-Za-z]', author))
        normalized_author = self._normalize_author_year_author(author, is_chinese=not has_english)
        if not normalized_author:
            return ""
        if has_english:
            return f"{normalized_author} ({year}{suffix})"
        return f"{normalized_author}（{year}{suffix}）"

    def _is_ai_citation_supported_by_body(self, citation_text: str, body_text: str) -> bool:
        from core.checker.citation_checking.reference_mapper import extract_author_year_from_citation

        author, year = extract_author_year_from_citation(citation_text or "")
        if not author or not year:
            return False

        normalized_author = re.sub(r"\s+", " ", author).strip()
        normalized_year = str(year).strip().lower()
        if not normalized_author or not normalized_year:
            return False

        compact_body = re.sub(r"\s+", "", body_text or "")
        normalized_body = re.sub(r"\s+", " ", body_text or "").replace("（", "(").replace("）", ")")
        year_pattern = re.escape(normalized_year)

        if re.search(r"[\u4e00-\u9fff]", normalized_author):
            compact_author = re.sub(r"\s+", "", normalized_author)
            variants = {
                compact_author,
                compact_author.replace("等", ""),
            }
            for variant in variants:
                if not variant:
                    continue
                explicit_pattern = rf"(?<![\u4e00-\u9fffA-Za-z0-9]){re.escape(variant)}(?:等)?[（(]{year_pattern}[）)]"
                parenthetical_pattern = rf"[（(]{re.escape(variant)}(?:等)?[，,]?\s*{year_pattern}[）)]"
                if re.search(explicit_pattern, compact_body) or re.search(parenthetical_pattern, compact_body):
                    return True
            return False

        for variant in self._build_english_author_lookup_variants(normalized_author):
            variant_pattern = re.escape(variant).replace(r"\ ", r"\s+")
            explicit_pattern = rf"(?<![A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]){variant_pattern}\s*[（(]\s*{year_pattern}\s*[）)]"
            parenthetical_pattern = rf"[（(]\s*{variant_pattern}\s*[,，]\s*{year_pattern}\s*[）)]"
            if re.search(explicit_pattern, normalized_body, flags=re.IGNORECASE):
                return True
            if re.search(parenthetical_pattern, normalized_body, flags=re.IGNORECASE):
                return True
        return False

    def _build_english_author_lookup_variants(self, author: str) -> set:
        normalized = re.sub(r"\s+", " ", author or "").strip()
        if not normalized:
            return set()

        variants = {
            normalized,
            normalized.replace(" and ", " & "),
            normalized.replace(" & ", " and "),
            re.sub(r"(?i)\bet\s+al\.?\b", "et al.", normalized),
            re.sub(r"(?i)\bet\s+al\.?\b", "et al", normalized),
        }
        return {variant.strip() for variant in variants if variant and variant.strip()}

    def _looks_like_standalone_latin_surname(self, text: str) -> bool:
        stripped = (text or "").strip()
        return bool(re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’\-]{2,}$", stripped))

    def _looks_like_latin_surname_continuation(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if re.match(r'^[A-Z](?:\.|\b)(?:\s+[A-Z](?:\.|\b)){0,6}\s+', stripped):
            return True
        if re.match(r'^[A-Z][a-z]+(?:\s+[a-z][a-z]+){1,6}', stripped):
            return True
        return False

    def _should_merge_with_current_reference(self, current_lines: list, next_line: str) -> bool:
        if not current_lines:
            return False
        last_line = (current_lines[-1] or "").strip()
        if not self._looks_like_standalone_latin_surname(last_line):
            return False
        return self._looks_like_latin_surname_continuation(next_line)

    def _merge_split_reference_fragments(self, parts: list) -> list:
        merged = []
        i = 0
        while i < len(parts):
            current = parts[i]
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if self._looks_like_standalone_chinese_name(current) and not self._looks_like_unnumbered_reference_start(nxt):
                    merged.append(f"{current} {nxt}".strip())
                    i += 2
                    continue
                if self._should_merge_conference_continuation(current, nxt):
                    merged.append(f"{current} {nxt}".strip())
                    i += 2
                    continue
                if self._looks_like_doc_type_tail(current) and not self._looks_like_unnumbered_reference_start(nxt):
                    merged.append(f"{current} {nxt}".strip())
                    i += 2
                    continue
                if self._looks_like_standalone_latin_surname(current) and self._looks_like_latin_surname_continuation(nxt):
                    merged.append(f"{current} {nxt}".strip())
                    i += 2
                    continue
            merged.append(current)
            i += 1
        return merged

    def _looks_like_doc_type_tail(self, text: str) -> bool:
        stripped = (text or "").strip()
        return bool(re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]\.\s*$', stripped))

    def _looks_like_standalone_chinese_name(self, text: str) -> bool:
        stripped = (text or "").strip()
        return bool(re.match(r'^[\u4e00-\u9fff]{2,4}\.$', stripped))

    def _should_merge_conference_continuation(self, current: str, nxt: str) -> bool:
        current_text = (current or "").strip()
        next_text = (nxt or "").strip()
        if not current_text or not next_text:
            return False

        if not re.search(r'[\[［]\s*C(?:/[A-Za-z]+)?\s*[\]］]\.\s*$', current_text, flags=re.IGNORECASE):
            return False

        if not re.match(
            r'^[\u4e00-\u9fff]{2,20}(?:学会|协会|大学|学院|研究院|研究所|出版社|集团|公司|中心|委员会|委|部|局|司)\.',
            next_text,
        ):
            return False

        if re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]', next_text[:40]):
            return False
        return True

    def _is_valid_reference(self, text: str) -> bool:
        """验证是否为有效的参考文献条目"""
        # 基本长度要求
        if len(text) < 10:
            return False

        is_numbered = bool(re.match(r'^\s*\[\d+\]', text))
        has_doc_type_marker = bool(re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]', text))

        # 检查是否包含必要的学术元素（编号条目允许跨行导致首行无年份）
        # 对无编号条目，若存在文献类型标识（如 [J/OL]）则允许进入后续规则判定。
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', text))
        if not has_year and not is_numbered and not has_doc_type_marker:
            return False

        # 结构化强信号：文献类型、DOI/URL、卷期页/页码定位等
        has_locator = bool(re.search(
            r'(?:\b\d+\(\d+\)\s*[:：]\s*[A-Za-z]?\d+|[:：]\s*\d+\s*[-–]\s*\d+|'
            r'\bVol\.?\s*\d+|\bNo\.?\s*\d+|\bpp?\.?\s*\d+|doi\s*[:：]|https?://)',
            text,
            re.IGNORECASE
        ))

        # 仅排除强语义结束段，避免误杀“时间表/图景/投入产出表”等真实文献标题
        exclude_keywords = ['附录', '致谢', '作者简历', 'Acknowledgements', 'Appendix']
        for keyword in exclude_keywords:
            if keyword in text[:50]:  # 检查前50个字符
                return False

        # 排除正文叙述型长句（高频误判来源）
        sentence_punct_count = text.count('。') + text.count('！') + text.count('？') + text.count('；') + text.count(';')
        if len(text) >= 120 and sentence_punct_count >= 1 and not is_numbered and not has_doc_type_marker and not has_locator:
            return False

        # 无强信号时，要求至少具备“作者.题名.来源, 年份, 卷(期):页码”这类结构
        has_plain_reference_shape = bool(re.search(r'\.\s*[^.]{2,120}\.\s*[^,，]{2,120}[,，]\s*(?:19|20)\d{2}', text))
        if not (is_numbered or has_doc_type_marker or has_locator or has_plain_reference_shape):
            return False

        return True

    def _remove_duplicates_and_validate(self, references: list) -> list:
        """移除重复并验证参考文献"""
        seen_texts = set()
        seen_text_list = []
        unique_refs = []

        for ref in references:
            # 标准化文本以进行比较（移除多余的空白字符）
            normalized_text = ' '.join(ref.text.split())

            if normalized_text in seen_texts:
                continue
            if self._looks_like_reference_tail_fragment(normalized_text):
                if any(existing.endswith(normalized_text) and len(existing) > len(normalized_text) + 8 for existing in seen_text_list):
                    continue

            seen_texts.add(normalized_text)
            seen_text_list.append(normalized_text)
            # 验证参考文献是否有效
            if self._is_valid_reference(ref.text):
                unique_refs.append(ref)

        return unique_refs

    def _looks_like_reference_tail_fragment(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if re.match(r'^[A-Z](?:\.)\s+[A-Z][a-z]', stripped):
            return True
        if re.match(r'^(?:19|20)\d{2}\s*[,，]\s*\(\d+\)\s*[:：]\s*\d+', stripped):
            return True
        return False
    
    def _is_reference_entry(self, text: str) -> bool:
        """判断是否为参考文献条目"""
        # 检查是否包含年份和基本结构
        has_year = bool(re.search(r'\d{4}', text))
        is_numbered = bool(re.match(r'^\s*\[\d+\]', text))
        has_basic_structure = len(text) > 10 and ('.' in text or '[' in text)
        
        # 排除非参考文献的条目
        exclude_keywords = ['附录', '致谢', '作者简历', '目录']
        has_exclude_keyword = any(keyword in text for keyword in exclude_keywords)
        
        return (has_year or is_numbered) and has_basic_structure and not has_exclude_keyword
    
    def _is_end_marker(self, text: str) -> bool:
        """判断是否为参考文献部分结束标记"""
        end_keywords = ['附录', '致谢', '作者简历', 'Appendix', 'Acknowledgements']
        if any(keyword in text for keyword in end_keywords) and len(text) < 80:
            return True
        # 某些 PDF 会把“致谢”拆成两行
        if text.strip() in {'致', '谢'}:
            return True
        return False

    def _is_toc_entry(self, text: str) -> bool:
        """判断是否为目录行（标题 + 页码）"""
        stripped = text.strip()
        return bool(re.search(r'\t\s*\d+\s*$', stripped) or re.search(r'\.{2,}\s*\d+\s*$', stripped))

    def _find_reference_start(self, paragraphs: list) -> int:
        """定位参考文献起点，优先文档后半段命中。"""
        if not paragraphs:
            return -1

        exact_candidates = []
        contains_candidates = []
        for i, paragraph in enumerate(paragraphs):
            text = (paragraph or "").strip()
            low = text.lower()
            if low in {'参考文献', '# 参考文献', 'references', '# references'}:
                exact_candidates.append(i)
            elif ('参考文献' in text or 'references' in low) and not self._is_toc_entry(text):
                contains_candidates.append(i)

        def pick_best(candidates: list) -> int:
            if not candidates:
                return -1
            scored = [(self._score_reference_start_candidate(paragraphs, idx), idx) for idx in candidates]
            best_score = max(score for score, _ in scored)
            best_indexes = [idx for score, idx in scored if score == best_score]
            return min(best_indexes)

        if exact_candidates:
            return pick_best(exact_candidates)

        if contains_candidates:
            return pick_best(contains_candidates)

        return -1

    def _score_reference_start_candidate(self, paragraphs: list, start_idx: int) -> int:
        """给参考文献起点候选打分，分值越高越可能是真实起点。"""
        score = 0
        window_end = min(len(paragraphs), start_idx + 200)

        for i in range(start_idx + 1, window_end):
            text = (paragraphs[i] or "").strip()
            if not text:
                continue
            if self._is_end_marker(text):
                break
            if self._is_toc_entry(text):
                continue
            if re.fullmatch(r'\d{1,4}', text):
                continue

            if re.match(r'^\s*\[\d+\]', text):
                score += 5
                continue

            if self._looks_like_unnumbered_reference_start(text):
                score += 4
            if re.search(r'\b(19|20)\d{2}\b', text):
                score += 1
            if re.search(r'[\[［]\s*[A-Za-z]+(?:/[A-Za-z]+)?\s*[\]］]', text):
                score += 2

        return score
    
    def _extract_doi(self, text: str) -> str:
        """提取DOI"""
        doi_pattern = r'doi:\s*([^\s,.;\)]+)|DOI:\s*([^\s,.;\)]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) or match.group(2)
        return None
    
    def _extract_url(self, text: str) -> str:
        """提取URL"""
        url_pattern = r'https?://[^\s,.;\)]+'
        match = re.search(url_pattern, text)
        return match.group(0) if match else None
