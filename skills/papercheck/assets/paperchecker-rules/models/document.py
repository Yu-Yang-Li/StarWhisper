from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Document:
    """文档数据模型"""
    content: List[str]  # 文档正文内容
    tables: List[str]   # 表格内容
    citations: List['Citation']  # 引用列表
    references: List['Reference']  # 参考文献列表
    metadata: Dict[str, Any]  # 文档元数据

@dataclass
class Citation:
    """引文数据模型"""
    text: str
    format_type: str  # 'number' 或 'author_year'
    context: str = ""
    author: Optional[str] = None
    year: Optional[str] = None

@dataclass
class Reference:
    """参考文献数据模型"""
    text: str
    author: Optional[str] = None
    year: Optional[str] = None