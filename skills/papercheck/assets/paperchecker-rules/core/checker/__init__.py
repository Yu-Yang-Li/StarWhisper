"""
PaperChecker 检查器模块
包含各种检查功能
"""
from .relevance_checker import RelevanceChecker
from .citation_checker import CitationChecker
from .consistency_checker import ConsistencyChecker
from .format_checker import FormatChecker

__all__ = [
    'RelevanceChecker',
    'CitationChecker',
    'ConsistencyChecker',
    'FormatChecker'
]