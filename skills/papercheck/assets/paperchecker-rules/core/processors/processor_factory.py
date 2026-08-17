from .citation_processor import CitationProcessor
from .document_processor import DocumentProcessor
from .batch_processor import BatchProcessor
from typing import Union, Dict, Any

class ProcessorFactory:
    """处理器工厂 - 创建不同类型的处理器"""
    
    @staticmethod
    def create_processor(processor_type: str, **kwargs) -> Union[CitationProcessor, DocumentProcessor, BatchProcessor]:
        """创建处理器实例"""
        if processor_type == 'citation':
            return CitationProcessor(config=kwargs.get('config'))
        elif processor_type == 'document':
            return DocumentProcessor()
        elif processor_type == 'batch':
            return BatchProcessor(max_workers=kwargs.get('max_workers', 4))
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
    
    @staticmethod
    def create_default_citation_processor() -> CitationProcessor:
        """创建默认引用处理器"""
        return CitationProcessor()
    
    @staticmethod
    def create_default_batch_processor() -> BatchProcessor:
        """创建默认批量处理器"""
        return BatchProcessor()