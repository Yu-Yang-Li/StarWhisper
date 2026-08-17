from abc import ABC, abstractmethod
from typing import Dict, Any
from models.document import Document

class BaseExtractor(ABC):
    """文档提取器基类"""
    
    @abstractmethod
    def extract(self, file_path: str) -> Document:
        """从文件路径提取文档内容"""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """验证文件格式是否支持"""
        pass