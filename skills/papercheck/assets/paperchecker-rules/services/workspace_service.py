from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import UploadFile

from services.exceptions import ServiceAuthError, ServiceValidationError
from utils.file_handler import save_upload_file

SUPPORTED_UPLOAD_EXTENSIONS = {".doc", ".docx", ".pdf"}


class WorkspaceService:
    def __init__(
        self,
        *,
        project_root: str,
        temp_dir: str,
        auth_enabled: Union[bool, Callable[[], bool]],
        user_resolver: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.temp_dir = Path(temp_dir).resolve()
        self.auth_enabled = auth_enabled
        self.user_resolver = user_resolver
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _is_auth_enabled(self) -> bool:
        if callable(self.auth_enabled):
            return bool(self.auth_enabled())
        return bool(self.auth_enabled)

    def _resolve_user_id(self, *, request: Any = None, user_id: Optional[str] = None) -> Optional[str]:
        if user_id:
            return str(user_id)

        if not self._is_auth_enabled():
            return None

        if self.user_resolver is None:
            raise ServiceAuthError("未登录或登录已过期")

        user = self.user_resolver(request)
        if not user:
            raise ServiceAuthError("未登录或登录已过期")

        uid = getattr(user, "user_id", None)
        if uid is None and isinstance(user, dict):
            uid = user.get("user_id")
        if uid is None:
            raise ServiceAuthError("未登录或登录已过期")
        return str(uid)

    def get_user_scoped_dir(self, *, request: Any = None, user_id: Optional[str] = None) -> Path:
        resolved_user_id = self._resolve_user_id(request=request, user_id=user_id)
        if resolved_user_id is None:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            return self.temp_dir

        scoped = (self.temp_dir / "users" / resolved_user_id).resolve()
        scoped.mkdir(parents=True, exist_ok=True)
        return scoped

    def resolve_scoped_path(
        self,
        file_path: str,
        *,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Path:
        if not file_path:
            raise ServiceValidationError("文件路径不能为空")

        allowed_base_dir = self.get_user_scoped_dir(request=request, user_id=user_id).resolve()
        candidate = Path(file_path)
        if candidate.is_absolute():
            abs_file_path = candidate.resolve()
        else:
            abs_file_path = (self.project_root / candidate).resolve()

        try:
            common = Path(os.path.commonpath([str(allowed_base_dir), str(abs_file_path)]))
        except ValueError as exc:
            raise ServiceValidationError("Invalid file path") from exc

        if common != allowed_base_dir:
            raise ServiceValidationError("Invalid file path")

        return abs_file_path

    def validate_upload(self, file: UploadFile, *, max_file_size: int) -> Dict[str, Any]:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size <= 0:
            raise ServiceValidationError(
                "上传文件为空（0 字节）。请重新选择可正常打开的 .docx、.doc 或 .pdf 文件。"
            )

        if file_size > max_file_size:
            raise ServiceValidationError(
                f"文件过大，最大支持 {max_file_size // (1024 * 1024)}MB"
            )

        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ServiceValidationError(
                f"不支持的文件类型: {ext}. 支持的类型: .docx, .doc, .pdf"
            )

        return {"file_size": file_size, "file_ext": ext}

    def save_upload(
        self,
        file: UploadFile,
        *,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Path:
        scoped_dir = self.get_user_scoped_dir(request=request, user_id=user_id)
        return Path(save_upload_file(file, str(scoped_dir))).resolve()

    def list_files(
        self,
        *,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scoped_dir = self.get_user_scoped_dir(request=request, user_id=user_id)
        files: List[Dict[str, Any]] = []

        for filename in sorted(os.listdir(scoped_dir)):
            file_path = scoped_dir / filename
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
                continue

            files.append(
                {
                    "name": filename,
                    "path": str(file_path.resolve()),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }
            )

        return files

    def delete_file(
        self,
        file_path: str,
        *,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_path = self.resolve_scoped_path(file_path, request=request, user_id=user_id)
        if not safe_path.exists():
            raise ServiceValidationError("文件不存在")

        safe_path.unlink()
        return {"status": "success", "message": "文件删除成功", "deleted_path": str(safe_path)}

    def clear_files(
        self,
        *,
        request: Any = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        files = self.list_files(request=request, user_id=user_id)
        deleted_count = 0
        for item in files:
            path = self.resolve_scoped_path(item["path"], request=request, user_id=user_id)
            if path.exists():
                path.unlink()
                deleted_count += 1
        return {"status": "success", "deleted_count": deleted_count}
