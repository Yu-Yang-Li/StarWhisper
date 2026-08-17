from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import docx


class DocConversionError(RuntimeError):
    """Raised when legacy .doc -> .docx conversion cannot be completed."""


@dataclass
class PreparedWordSource:
    path: Path
    source_type: str
    temp_dir: Optional[Path] = None
    conversion_tool: Optional[str] = None

    def cleanup(self) -> None:
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def prepare_word_source(file_path: str) -> PreparedWordSource:
    source = Path(file_path).resolve()
    suffix = source.suffix.lower()
    if suffix not in {".doc", ".docx", ".docm"}:
        raise DocConversionError(f"不支持的 Word 文件类型: {suffix}")
    if suffix == ".doc":
        return _convert_source_to_docx(source, source_type="doc")
    if _is_docx_readable(source):
        return PreparedWordSource(path=source, source_type=suffix.lstrip("."))
    return _convert_source_to_docx(source, source_type=suffix.lstrip("."))


def _convert_source_to_docx(source: Path, source_type: str) -> PreparedWordSource:
    work_dir = Path(tempfile.mkdtemp(prefix="paperchecker_doc_convert_"))
    target = work_dir / f"{source.stem}.docx"
    errors: List[str] = []

    if shutil.which("textutil"):
        ok, detail = _run_textutil(source, target)
        if ok and _is_docx_readable(target):
            return PreparedWordSource(
                path=target,
                source_type=source_type,
                temp_dir=work_dir,
                conversion_tool="textutil",
            )
        errors.append(detail)
    else:
        errors.append("textutil 不可用")

    for bin_name in ("libreoffice", "soffice"):
        if not shutil.which(bin_name):
            continue
        ok, detail = _run_libreoffice(bin_name, source, target, work_dir)
        if ok and _is_docx_readable(target):
            return PreparedWordSource(
                path=target,
                source_type=source_type,
                temp_dir=work_dir,
                conversion_tool=bin_name,
            )
        errors.append(detail)

    available = [name for name in ("textutil", "libreoffice", "soffice") if shutil.which(name)]
    hint = (
        "未检测到可用转换工具（需 textutil 或 libreoffice/soffice）。"
        if not available
        else f"已尝试工具: {', '.join(available)}。"
    )
    detail = " | ".join(e for e in errors if e)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise DocConversionError(f"{source.suffix} 转换失败: {hint} {detail}".strip())


def _run_textutil(source: Path, target: Path) -> tuple[bool, str]:
    cmd = ["textutil", "-convert", "docx", "-output", str(target), str(source)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        return False, f"textutil 失败: {completed.stderr.strip() or completed.stdout.strip()}"
    if not target.exists():
        return False, "textutil 执行成功但未生成 docx 文件"
    return True, "ok"


def _run_libreoffice(bin_name: str, source: Path, target: Path, out_dir: Path) -> tuple[bool, str]:
    cmd = [bin_name, "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(source)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        return False, f"{bin_name} 失败: {completed.stderr.strip() or completed.stdout.strip()}"
    if not target.exists():
        fallback = out_dir / f"{source.stem}.docx"
        if fallback.exists() and fallback != target:
            fallback.replace(target)
    if not target.exists():
        return False, f"{bin_name} 执行成功但未生成 docx 文件"
    return True, "ok"


def _is_docx_readable(path: Path) -> bool:
    try:
        docx.Document(str(path))
        return True
    except Exception:
        return False
