from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from services.exceptions import ServiceValidationError  # noqa: E402
from services.workspace_service import WorkspaceService  # noqa: E402
from utils.file_handler import save_upload_file  # noqa: E402


def make_upload(filename: str, payload: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(payload), filename=filename)


def test_workspace_upload_rejects_zero_byte_file(tmp_path):
    service = WorkspaceService(
        project_root=str(tmp_path),
        temp_dir=str(tmp_path / "uploads"),
        auth_enabled=False,
    )

    with pytest.raises(ServiceValidationError, match="0 字节"):
        service.validate_upload(make_upload("empty.pdf", b""), max_file_size=1024)


def test_save_upload_file_rejects_zero_byte_file(tmp_path):
    with pytest.raises(HTTPException) as exc_info:
        save_upload_file(make_upload("empty.pdf", b""), str(tmp_path))

    assert exc_info.value.status_code == 400
    assert "0 字节" in str(exc_info.value.detail)
    assert list(tmp_path.iterdir()) == []


def test_save_upload_file_preserves_non_empty_pdf_size(tmp_path):
    payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    saved = Path(save_upload_file(make_upload("paper.pdf", payload), str(tmp_path)))

    assert saved.exists()
    assert saved.stat().st_size == len(payload)
    assert saved.read_bytes() == payload


def test_analysis_report_endpoint_rejects_zero_byte_pdf():
    client = TestClient(app)

    response = client.post(
        "/api/v2/analysis/report",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["run"]["status"] == "failed"
    assert "0 字节" in payload["error"]["message"]


def test_workspace_upload_endpoint_reports_saved_size_for_non_empty_pdf():
    client = TestClient(app)
    payload = b"%PDF-1.4\n%%EOF\n"

    response = client.post(
        "/api/v2/workspace/upload",
        files={"file": ("paper.pdf", payload, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    saved = body["evidence"]["file"]
    assert saved["size"] == len(payload)

    client.delete("/api/v2/workspace/file", params={"file_path": saved["path"]})
