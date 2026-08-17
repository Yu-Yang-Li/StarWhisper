from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class CheckType(Enum):
    """检查类型"""
    CITATIONS = "citations"      # 引用检查
    RELEVANCE = "relevance"      # 相关性检查
    # 为未来扩展预留

@dataclass
class ComplianceResult:
    """合规性检查结果"""
    check_type: CheckType               # 检查类型
    is_compliant: bool                  # 是否合规
    issues: List[Dict[str, Any]]        # 发现的问题（使用字典表示）
    statistics: Dict[str, Any]          # 统计信息
    metadata: Dict[str, Any]            # 额外元数据