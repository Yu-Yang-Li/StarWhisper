from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile

from services.exceptions import ServiceValidationError
from services.report_service import analyze_document
from services.workspace_service import SUPPORTED_UPLOAD_EXTENSIONS, WorkspaceService
from utils.file_handler import cleanup_file


class AnalysisService:
    def __init__(self, workspace_service: WorkspaceService, *, max_file_size: int) -> None:
        self.workspace_service = workspace_service
        self.max_file_size = max_file_size

    def analyze_upload(
        self,
        *,
        file: UploadFile,
        author_format: str,
        citation_standard: str = "legacy",
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.workspace_service.validate_upload(file, max_file_size=self.max_file_size)

        saved_file_path: Optional[Path] = None
        try:
            saved_file_path = self.workspace_service.save_upload(file, request=request, user_id=user_id)
            result = analyze_document(
                str(saved_file_path),
                author_format=author_format,
                citation_standard=citation_standard,
            )
            return result["raw_report"]
        finally:
            if saved_file_path:
                cleanup_file(str(saved_file_path))

    def analyze_path(
        self,
        *,
        file_path: str,
        author_format: str,
        citation_standard: str = "legacy",
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_path = self.workspace_service.resolve_scoped_path(file_path, request=request, user_id=user_id)
        if not safe_path.exists():
            raise ServiceValidationError("File not found")

        if safe_path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ServiceValidationError(
                f"不支持的文件类型: {safe_path.suffix}. 支持的类型: .docx, .doc, .pdf"
            )

        if safe_path.stat().st_size <= 0:
            raise ServiceValidationError("文件为空（0 字节），请删除后重新上传原始文件")

        result = analyze_document(
            str(safe_path),
            author_format=author_format,
            citation_standard=citation_standard,
        )
        return result["raw_report"]
