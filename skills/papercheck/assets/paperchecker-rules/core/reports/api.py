from .report_generator import ReportGenerator
from .storage import ReportStorage
from models.report import BaseReport
from typing import Dict, Any, List
import os

class ReportService:
    """报告服务API"""
    
    def __init__(self, storage_dir: str = "reports"):
        self.storage = ReportStorage(storage_dir)
        self.generator = ReportGenerator()
    
    def create_report(self, analysis_result: Dict[str, Any], report_type: str = "comprehensive") -> str:
        """创建并保存报告"""
        if report_type == "citation":
            report = self.generator.generate_citation_report(analysis_result)
        elif report_type == "format":
            report = self.generator.generate_format_report(analysis_result)
        elif report_type == "consistency":
            report = self.generator.generate_consistency_report(analysis_result)
        elif report_type == "comprehensive":
            report = self.generator.generate_comprehensive_report(analysis_result)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        return self.storage.save_report(report)
    
    def list_available_reports(self) -> List[Dict[str, Any]]:
        """列出可用的报告"""
        report_files = self.storage.list_reports()
        reports_info = []
        
        for report_file in report_files:
            try:
                report_data = self.storage.load_report(report_file)
                reports_info.append({
                    'file_path': report_file,
                    'report_type': report_data.get('report_type', 'unknown'),
                    'generated_at': report_data.get('generated_at', 'unknown'),
                    'document_path': report_data.get('document_path', 'unknown')
                })
            except Exception:
                # 如果报告文件损坏，则跳过
                continue
        
        return reports_info
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """获取报告统计信息"""
        all_reports = self.storage.list_reports()
        stats = {
            'total_reports': len(all_reports),
            'by_type': {},
            'latest_report': None
        }
        
        for report_file in all_reports:
            try:
                report_data = self.storage.load_report(report_file)
                report_type = report_data.get('report_type', 'unknown')
                
                if report_type not in stats['by_type']:
                    stats['by_type'][report_type] = 0
                stats['by_type'][report_type] += 1
                
                if stats['latest_report'] is None or report_data.get('generated_at', '') > stats['latest_report']:
                    stats['latest_report'] = report_data.get('generated_at', '')
            except Exception:
                continue
        
        return stats