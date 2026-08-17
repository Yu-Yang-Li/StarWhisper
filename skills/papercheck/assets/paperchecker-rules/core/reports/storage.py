import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from models.report import BaseReport

class ReportStorage:
    """报告存储管理器"""
    
    def __init__(self, storage_dir: str = "reports"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
    
    def save_report(self, report: BaseReport, filename: str = None) -> str:
        """保存报告到文件"""
        if filename is None:
            filename = f"{report.report_type}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = os.path.join(self.storage_dir, filename)
        
        # 将报告序列化并保存
        import json
        from dataclasses import asdict
        
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        report_dict = asdict(report)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, default=serialize_datetime, ensure_ascii=False, indent=2)
        
        return file_path
    
    def load_report(self, file_path: str) -> Dict[str, Any]:
        """从文件加载报告"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_reports(self) -> List[str]:
        """列出所有报告文件"""
        reports = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                reports.append(os.path.join(self.storage_dir, filename))
        return reports
    
    def search_reports(self, **criteria) -> List[str]:
        """根据条件搜索报告"""
        matching_reports = []
        
        for report_file in self.list_reports():
            report_data = self.load_report(report_file)
            
            # 检查搜索条件
            match = True
            for key, value in criteria.items():
                if key not in report_data or report_data[key] != value:
                    match = False
                    break
            
            if match:
                matching_reports.append(report_file)
        
        return matching_reports
    
    def get_recent_reports(self, count: int = 10) -> List[str]:
        """获取最近的报告"""
        all_reports = self.list_reports()
        # 按文件修改时间排序
        all_reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return all_reports[:count]