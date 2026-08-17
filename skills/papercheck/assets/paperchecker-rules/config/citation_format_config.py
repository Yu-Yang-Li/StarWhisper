"""
引用格式配置
定义不同的引用格式选项和默认配置
"""

from enum import Enum
from typing import Dict, Any


class CitationFormatType(Enum):
    """引用格式类型枚举"""
    CHINESE_ACADEMY_OF_SCIENCES = "chinese_academy_of_sciences"  # 中科院格式（作者+年份）
    APA = "apa"  # APA格式
    MLA = "mla"  # MLA格式
    CHICAGO = "chicago"  # 芝加哥格式
    NUMERIC = "numeric"  # 数字格式


class CitationFormatConfig:
    """引用格式配置类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化引用格式配置
        
        Args:
            config: 配置字典
        """
        if config is None:
            config = {}
        
        # 获取引用格式类型，默认为中科院格式
        format_type = config.get("citation_format_type", "chinese_academy_of_sciences")
        self.format_type = CitationFormatType(format_type)
        
        # 默认使用中科院格式
        if not hasattr(self, 'format_type') or self.format_type is None:
            self.format_type = CitationFormatType.CHINESE_ACADEMY_OF_SCIENCES
    
    @property
    def is_chinese_academy_of_sciences_format(self) -> bool:
        """是否使用中科院格式"""
        return self.format_type == CitationFormatType.CHINESE_ACADEMY_OF_SCIENCES
    
    @property
    def is_apa_format(self) -> bool:
        """是否使用APA格式"""
        return self.format_type == CitationFormatType.APA
    
    @property
    def is_mla_format(self) -> bool:
        """是否使用MLA格式"""
        return self.format_type == CitationFormatType.MLA
    
    @property
    def is_chicago_format(self) -> bool:
        """是否使用芝加哥格式"""
        return self.format_type == CitationFormatType.CHICAGO
    
    @property
    def is_numeric_format(self) -> bool:
        """是否使用数字格式"""
        return self.format_type == CitationFormatType.NUMERIC


def get_default_citation_format_config() -> CitationFormatConfig:
    """获取默认的引用格式配置（中科院格式）"""
    return CitationFormatConfig({
        "citation_format_type": "chinese_academy_of_sciences"
    })