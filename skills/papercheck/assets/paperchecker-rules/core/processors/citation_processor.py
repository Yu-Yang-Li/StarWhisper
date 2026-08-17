from .base_processor import BaseProcessor
from core.extractor.extractor_factory import ExtractorFactory
from core.checker.checker_factory import CheckerFactory
from models.document import Document
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CitationProcessor(BaseProcessor):
    """引用处理器核心 - 全新的重构架构"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.extractor_factory = ExtractorFactory()
        self.checker_factory = CheckerFactory()
    
    def process(
        self,
        file_path: str,
        author_format: str = "full",
        citation_standard: str = "legacy",
    ) -> Dict[str, Any]:
        """
        使用全新的重构架构进行处理
        1. 使用Extractor提取文档内容
        2. 使用Checker进行引用匹配分析
        """
        start_time = datetime.now()
        
        try:
            # 步骤1: 使用Extractor提取文档内容
            extractor = self.extractor_factory.get_extractor(file_path)
            logger.info(f"开始提取文档: {file_path}")
            document = extractor.extract(file_path)
            logger.info(f"文档提取完成，共提取 {len(document.citations)} 个引用，{len(document.references)} 个参考文献")
            
            # 步骤2: 使用Checker执行引用分析
            from core.processors.compliance_processor import ComplianceProcessor
            from models.compliance import CheckType
            checker = self.checker_factory.get_checker(CheckType.CITATIONS)
            compliance_processor = ComplianceProcessor([checker])
            
            # 使用合规性处理器进行检查
            analysis_result = compliance_processor.process(
                file_path,
                document=document,
                checker_kwargs={
                    "author_format": author_format,
                    "citation_standard": citation_standard,
                },
            )
            
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"引用处理完成，耗时 {duration_ms}ms")
            return analysis_result  # 直接返回合规性处理器的结果
            
        except Exception as e:
            logger.error(f"处理文档时发生错误: {str(e)}")
            raise
