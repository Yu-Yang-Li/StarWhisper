from models.report import CitationReport, FormatReport, ConsistencyReport, ComprehensiveReport
from models.document import Document
from typing import Dict, Any, Optional
import json
from datetime import datetime

class ReportGenerator:
    """报告生成器 - 根据处理结果生成各种格式的报告"""
    
    @staticmethod
    def generate_citation_report(analysis_result: Dict[str, Any]) -> CitationReport:
        """生成引用分析报告"""
        return CitationReport(
            report_id=f"citation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            document_path=analysis_result.get("document", ""),
            total_citations=analysis_result.get("citation_analysis", {}).get("total_citations", 0),
            total_references=analysis_result.get("citation_analysis", {}).get("total_references", 0),
            matched_citations=analysis_result.get("citation_analysis", {}).get("matched_citations", 0),  # 使用兼容的键名
            unmatched_citations=analysis_result.get("citation_analysis", {}).get("unmatched_citations", 0),  # 使用兼容的键名
            corrections_needed=analysis_result.get("citation_analysis", {}).get("corrections_needed", 0),  # 使用兼容的键名
            unused_references=analysis_result.get("citation_analysis", {}).get("unused_references_count", 0),  # 使用兼容的键名
            details=analysis_result.get("citation_analysis", {})
        )
    
    @staticmethod
    def generate_format_report(analysis_result: Dict[str, Any]) -> FormatReport:
        """生成格式检查报告"""
        format_analysis = analysis_result.get("format_analysis", {})
        issues = format_analysis.get("issues", [])
        
        return FormatReport(
            report_id=f"format_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            document_path=analysis_result.get("document", ""),
            total_issues=len(issues),
            issues=issues
        )
    
    @staticmethod
    def generate_consistency_report(analysis_result: Dict[str, Any]) -> ConsistencyReport:
        """生成一致性检查报告"""
        consistency_analysis = analysis_result.get("consistency_analysis", {})
        inconsistencies = consistency_analysis.get("inconsistencies", [])
        
        return ConsistencyReport(
            report_id=f"consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            document_path=analysis_result.get("document", ""),
            total_inconsistencies=len(inconsistencies),
            inconsistencies=inconsistencies
        )
    
    @staticmethod
    def generate_comprehensive_report(analysis_results: Dict[str, Any]) -> ComprehensiveReport:
        """生成综合报告"""
        # 生成各个子报告
        citation_report = None
        format_report = None
        consistency_report = None
        
        if "citation_analysis" in analysis_results:
            citation_report = ReportGenerator.generate_citation_report(analysis_results)
        
        if "format_analysis" in analysis_results:
            format_report = ReportGenerator.generate_format_report(analysis_results)
        
        if "consistency_analysis" in analysis_results:
            consistency_report = ReportGenerator.generate_consistency_report(analysis_results)
        
        # 生成汇总信息
        summary = analysis_results.get("overall_summary", {})
        
        return ComprehensiveReport(
            report_id=f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            document_path=analysis_results.get("document", ""),
            citation_report=citation_report,
            format_report=format_report,
            consistency_report=consistency_report,
            summary=summary
        )