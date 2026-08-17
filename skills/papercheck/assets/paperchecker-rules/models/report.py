from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class BaseReport:
    """基础报告模型"""
    report_id: str
    generated_at: datetime
    report_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CitationReport:
    """引用分析报告模型（不从BaseReport继承以避免参数顺序问题）"""
    report_id: str
    generated_at: datetime
    report_type: str
    document_path: str
    total_citations: int
    total_references: int
    matched_citations: int
    unmatched_citations: int
    corrections_needed: int
    unused_references: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.report_type = "citation_analysis"
        self.match_rate = f"{self.matched_citations/self.total_citations*100:.1f}%" if self.total_citations > 0 else "0%"

@dataclass
class FormatReport:
    """格式检查报告模型"""
    report_id: str
    generated_at: datetime
    report_type: str
    document_path: str
    total_issues: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.report_type = "format_analysis"

@dataclass
class ConsistencyReport:
    """一致性检查报告模型"""
    report_id: str
    generated_at: datetime
    report_type: str
    document_path: str
    total_inconsistencies: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.report_type = "consistency_analysis"

@dataclass
class ComprehensiveReport:
    """综合报告模型"""
    report_id: str
    generated_at: datetime
    report_type: str
    document_path: str
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation_report: Optional[CitationReport] = None
    format_report: Optional[FormatReport] = None
    consistency_report: Optional[ConsistencyReport] = None
    
    def __post_init__(self):
        self.report_type = "comprehensive_analysis"
        self.total_issues = 0
        if self.citation_report:
            self.total_issues += self.citation_report.unmatched_citations + self.citation_report.corrections_needed
        if self.format_report:
            self.total_issues += self.format_report.total_issues
        if self.consistency_report:
            self.total_issues += self.consistency_report.total_inconsistencies