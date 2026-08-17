from typing import Dict, Type, List
from core.checker.base_checker import BaseChecker
from core.checker.citation_checker import CitationChecker
from core.checker.relevance_checker import RelevanceChecker
from models.compliance import CheckType

class CheckerFactory:
    """检查器工厂 - 支持动态注册和获取检查器"""

    _checkers: Dict[CheckType, Type[BaseChecker]] = {
        CheckType.CITATIONS: CitationChecker,
        CheckType.RELEVANCE: RelevanceChecker,
        # 将来可以注册更多检查器
    }
    
    @classmethod
    def register_checker(cls, check_type: CheckType, checker_class: Type[BaseChecker]):
        """注册新的检查器"""
        cls._checkers[check_type] = checker_class
    
    @classmethod
    def get_checker(cls, check_type: CheckType) -> BaseChecker:
        """获取指定类型的检查器"""
        if check_type not in cls._checkers:
            raise ValueError(f"Unknown checker type: {check_type}")
        
        return cls._checkers[check_type]()
    
    @classmethod
    def get_all_checkers(cls) -> List[BaseChecker]:
        """获取所有检查器实例"""
        return [checker_class() for checker_class in cls._checkers.values()]
    
    @classmethod
    def get_available_check_types(cls) -> List[CheckType]:
        """返回可用的检查器类型"""
        return list(cls._checkers.keys())