from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.checker.relevance_checker import RelevanceChecker
from core.extractor.markdown_extractor import MarkdownExtractor
from core.extractor.word_extractor import WordExtractor
from services.exceptions import ServiceValidationError
from services.workspace_service import WorkspaceService


class RelevanceService:
    def __init__(self, workspace_service: WorkspaceService) -> None:
        self.workspace_service = workspace_service

    def _select_extractor(self, file_path: Path):
        ext = file_path.suffix.lower().lstrip(".")
        if ext in {"doc", "docx"}:
            return WordExtractor()
        if ext == "pdf":
            from core.extractor.pdf_extractor import PDFExtractor

            return PDFExtractor()
        if ext == "md":
            return MarkdownExtractor()
        raise ServiceValidationError(f"不支持的文件类型: {ext}")

    def _extract_document(self, file_path: Path):
        if not file_path.exists():
            raise ServiceValidationError("文件不存在")
        extractor = self._select_extractor(file_path)
        return extractor.extract(str(file_path))

    def extract_citations_by_path(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path).resolve()
        document = self._extract_document(path)
        citations: List[Dict[str, Any]] = []
        for citation in document.citations:
            citations.append(
                {
                    "text": citation.text,
                    "format_type": citation.format_type,
                    "context": citation.context,
                    "author": citation.author,
                    "year": citation.year,
                }
            )
        return citations

    def extract_citations(
        self,
        *,
        file_path: str,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        safe_path = self.workspace_service.resolve_scoped_path(file_path, request=request, user_id=user_id)
        return self.extract_citations_by_path(str(safe_path))

    def relevance_check_by_path(
        self,
        *,
        file_path: str,
        target_content: str,
        task_type: str,
        use_full_content: bool,
    ) -> Dict[str, Any]:
        path = Path(file_path).resolve()
        document = self._extract_document(path)

        checker = RelevanceChecker(use_full_content=use_full_content)
        result = checker.check(document, target_content, task_type)

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y年%m月%d日%H:%M:%S")

        return {
            "document": path.name,
            "generated_time": formatted_time,
            "task_type": task_type,
            "check_method": "accurate_check" if use_full_content else "quick_check",
            "relevance_score": result.statistics.get("relevance_score", 0),
            "is_suitable_for_citation": result.is_compliant,
            "brief_basis": result.issues[0].get("brief_basis", "") if result.issues else "",
            "detailed_reasoning": result.issues[0].get("detailed_reasoning", "") if result.issues else "",
            "raw_result": result.metadata.get("ai_response", ""),
        }

    def relevance_check(
        self,
        *,
        file_path: str,
        target_content: str,
        task_type: str,
        use_full_content: bool,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_path = self.workspace_service.resolve_scoped_path(file_path, request=request, user_id=user_id)
        return self.relevance_check_by_path(
            file_path=str(safe_path),
            target_content=target_content,
            task_type=task_type,
            use_full_content=use_full_content,
        )
