"""
论文引用检查器 - 完整的端到端处理类
负责从文档中提取内容到执行引用检查的完整流程
"""

import docx
import re
from typing import List, Dict, Any
import os
import json
from models.document import Document, Citation, Reference
from core.extractor.extractor_factory import ExtractorFactory
from core.checker.checker_factory import CheckerFactory


class DocumentCitationChecker:
    """
    文档引用检查器 - 提供完整的端到端引用检查功能
    从文档加载、内容提取到引用检查的完整流程
    """
    
    def __init__(self, doc_path: str, config_path: str = "config.json"):
        self.doc_path = doc_path
        self.config_path = config_path
        self.document = None
        self.check_results = None
        
    def extract_content(self):
        """
        从文档中提取内容（引用、参考文献等）
        """
        # 使用Extractor工厂获取适当的提取器
        extractor = ExtractorFactory.get_extractor(self.doc_path)
        
        # 提取文档内容
        self.document = extractor.extract(self.doc_path)
        
        return self.document
    
    def check_citations(self):
        """
        执行引用检查
        """
        if not self.document:
            self.extract_content()
        
        # 使用Checker工厂获取引用检查器
        checker = CheckerFactory.get_checker('citation')
        
        # 执行检查
        self.check_results = checker.check(self.document)
        
        return self.check_results
    
    def generate_report(self):
        """
        生成检查报告
        """
        if not self.check_results:
            self.check_citations()
        
        # 生成格式化的报告
        report = []
        report.append("<h1>论文引用合规性检查报告</h1>\n")
        
        report.append(f"<p>文档: {self.doc_path}</p>\n")
        report.append(f"<p>总引用数: {self.check_results['total_citations']}</p>\n")
        report.append(f"<p>总参考文献数: {self.check_results['total_references']}</p>\n")
        report.append(f"<p>匹配数: {self.check_results['matched_count']}</p>\n")
        report.append(f"<p>未匹配数: {self.check_results['unmatched_count']}</p>\n")
        report.append(f"<p>匹配率: {self.check_results['match_rate']}</p>\n")
        
        # 添加详细结果
        report.append("<h2>详细结果</h2>\n")
        for result in self.check_results['results']:
            if result['matched']:
                report.append(f"<p><strong>✓</strong> {result['original_citation']} -> {result['reference_text'][:100]}...</p>\n")
            else:
                report.append(f"<p><strong>✗</strong> {result['original_citation']} (未匹配)</p>\n")
        
        return "".join(report)


def create_standalone_checker(doc_path: str, config_path: str = "config.json"):
    """
    创建独立的引用检查器（用于向后兼容）
    """
    return DocumentCitationChecker(doc_path, config_path)