import os

from .base_extractor import BaseExtractor
from .word_extractor import WordExtractor
from .pdf_extractor import PDFExtractor
from .markdown_extractor import MarkdownExtractor
from typing import Dict, Type

class ExtractorFactory:
    """提取器工厂"""
    
    _extractors: Dict[str, Type[BaseExtractor]] = {
        'docx': WordExtractor,
        'doc': WordExtractor,
        'pdf': PDFExtractor,
        'md': MarkdownExtractor,
    }
    
    @classmethod
    def register_extractor(cls, file_ext: str, extractor_class: Type[BaseExtractor]):
        """注册新的提取器"""
        cls._extractors[file_ext] = extractor_class
    
    @classmethod
    def get_extractor(cls, file_path: str) -> BaseExtractor:
        """根据文件路径获取对应的提取器"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext not in cls._extractors:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return cls._extractors[ext]()
    
    @classmethod
    def supported_types(cls) -> list:
        """返回支持的文件类型"""
        return list(cls._extractors.keys())
