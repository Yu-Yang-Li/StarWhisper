"""Service layer for PaperChecker.

Keep package import lightweight. Heavy report dependencies are loaded lazily so
upload validation and workspace tests do not require the full PDF stack.
"""

__all__ = ["analyze_and_export", "analyze_document", "run_smoke_suite"]


def __getattr__(name):
    if name in __all__:
        from . import report_service

        return getattr(report_service, name)
    raise AttributeError(name)
