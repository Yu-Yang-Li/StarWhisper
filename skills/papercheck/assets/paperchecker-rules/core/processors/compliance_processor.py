from typing import List, Optional, Dict, Any
from models.document import Document
from core.checker.base_checker import BaseChecker
from models.compliance import ComplianceResult, CheckType

class ComplianceProcessor:
    """合规性处理器 - 协调Extractor和Checker"""
    
    def __init__(self, checkers: List[BaseChecker] = None):
        self.checkers = checkers or []
    
    def add_checker(self, checker: BaseChecker):
        """添加检查器"""
        self.checkers.append(checker)
    
    def process(
        self,
        file_path: str,
        document: Optional[Document] = None,
        checker_kwargs: Optional[Dict[str, Any]] = None
    ) -> dict:  # 返回兼容格式的字典
        """
        处理文档的合规性检查
        
        Args:
            file_path: 文档文件路径
            document: 可选的预提取文档对象（用于避免重复抽取）
            checker_kwargs: 传给 checker.check 的可选参数
            
        Returns:
            dict: 兼容前端的检查结果字典
        """
        checker_kwargs = checker_kwargs or {}

        # 1. 使用Extractor提取文档内容（如已传入 document 则复用）
        if document is None:
            from core.extractor.extractor_factory import ExtractorFactory
            extractor = ExtractorFactory.get_extractor(file_path)
            document = extractor.extract(file_path)
        
        # 2. 运行引用检查器并获取兼容输出
        for checker in self.checkers:
            if checker.get_check_type() == CheckType.CITATIONS:
                result = checker.check(document, **checker_kwargs)
                # 直接返回兼容格式的输出
                return result.metadata.get("compatibility_output", {})
        
        # 如果没有找到引用检查器，返回空结果
        return {
            "test_date": __import__('datetime').datetime.now().isoformat(),
            "document": file_path,
            "total_citations": 0,
            "total_references": 0,
            "results": [],
            "matched_count": 0,
            "unmatched_count": 0,
            "corrected_count": 0,
            "formatted_count": 0,
            "unused_references_count": 0,
            "match_rate": "0%",
            "corrections_needed": [],
            "formatting_needed": [],
            "unused_references": []
        }
