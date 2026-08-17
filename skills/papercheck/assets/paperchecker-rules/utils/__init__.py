"""PaperChecker utility package.

Keep package import lightweight. Import concrete submodules directly, for
example `from utils.file_handler import save_upload_file`.
"""

__all__ = [
    "validate_file_type",
    "save_upload_file",
    "cleanup_file",
    "MineruPDFToMD",
    "convert_pdf_to_markdown",
    "fix_title_levels",
    "build_markdown_report",
    "save_markdown_report",
    "save_markdown_as_pdf",
]


def __getattr__(name):
    if name in {"validate_file_type", "save_upload_file", "cleanup_file"}:
        from . import file_handler

        return getattr(file_handler, name)
    if name in {"MineruPDFToMD", "convert_pdf_to_markdown", "fix_title_levels"}:
        from . import mineru_pdf_converter

        return getattr(mineru_pdf_converter, name)
    if name in {"build_markdown_report", "save_markdown_report"}:
        from . import report_markdown

        return getattr(report_markdown, name)
    if name == "save_markdown_as_pdf":
        from . import report_pdf

        return getattr(report_pdf, name)
    raise AttributeError(name)
