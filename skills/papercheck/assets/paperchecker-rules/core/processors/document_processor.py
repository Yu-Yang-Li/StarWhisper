from .base_processor import BaseProcessor
from .citation_processor import CitationProcessor
from core.extractor.extractor_factory import ExtractorFactory
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor(BaseProcessor):
    """文档处理器 - 处理单个或多个文档的完整流程"""
    
    def __init__(self):
        self.extractor_factory = ExtractorFactory()
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """处理单个文档"""
        # 使用CitationProcessor进行处理
        processor = CitationProcessor()
        return processor.process(file_path)
    
    def process_batch(self, file_paths: List[str], config: Dict = None) -> List[Dict[str, Any]]:
        """批量处理多个文档"""
        results = []
        
        for file_path in file_paths:
            try:
                logger.info(f"处理文档: {file_path}")
                result = self.process(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文档 {file_path} 时发生错误: {str(e)}")
                results.append({
                    "document": file_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results