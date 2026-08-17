"""
引用格式化器
根据配置生成不同格式的引用
"""

from typing import List, Dict, Any, Optional
from config.citation_format_config import CitationFormatConfig, CitationFormatType


class CitationFormatter:
    """引用格式化器"""
    
    def __init__(self, config: CitationFormatConfig):
        """
        初始化引用格式化器
        
        Args:
            config: 引用格式配置
        """
        self.config = config
    
    def format_citation(
        self, 
        authors: List[str], 
        year: str, 
        title: Optional[str] = None,
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        doi: Optional[str] = None
    ) -> str:
        """
        根据配置格式化引用
        
        Args:
            authors: 作者列表
            year: 年份
            title: 标题
            journal: 期刊名
            volume: 卷号
            issue: 期号
            pages: 页码
            doi: DOI
            
        Returns:
            格式化后的引用字符串
        """
        if self.config.is_chinese_academy_of_sciences_format:
            return self._format_chinese_academy_of_sciences(authors, year)
        elif self.config.is_apa_format:
            return self._format_apa(authors, year, title, journal, volume, issue, pages)
        elif self.config.is_mla_format:
            return self._format_mla(authors, year, title, journal, volume, pages)
        elif self.config.is_chicago_format:
            return self._format_chicago(authors, year, title, journal, volume, issue, pages)
        elif self.config.is_numeric_format:
            return self._format_numeric(authors, year)
        else:
            # 默认使用中科院格式
            return self._format_chinese_academy_of_sciences(authors, year)
    
    def _format_chinese_academy_of_sciences(self, authors: List[str], year: str) -> str:
        """
        格式化为中科院格式（作者+年份）
        
        Args:
            authors: 作者列表
            year: 年份
            
        Returns:
            格式化后的引用字符串
        """
        if not authors:
            return f"（{year}）"
        
        # 判断是否是中文作者
        is_chinese = self._contains_chinese(authors[0]) if authors else False
        
        if is_chinese:
            # 中文作者
            if len(authors) > 1:
                return f"{authors[0]} 等（{year}）"
            else:
                return f"{authors[0]}（{year}）"
        else:
            # 英文作者，提取姓氏
            surnames = [self._extract_surname(author) for author in authors]
            
            # 处理机构或团体作者
            if len(surnames) == 1 and (len(surnames[0].split()) > 2 or surnames[0].isupper()):
                return f"{surnames[0]}（{year}）"
            
            # 根据作者数量格式化
            if len(surnames) > 2:
                return f"{surnames[0]} et al.（{year}）"
            elif len(surnames) == 2:
                return f"{surnames[0]} & {surnames[1]}（{year}）"
            else:
                return f"{surnames[0]}（{year}）"
    
    def _format_apa(
        self, 
        authors: List[str], 
        year: str, 
        title: Optional[str] = None,
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None
    ) -> str:
        """
        格式化为APA格式
        
        Args:
            authors: 作者列表
            year: 年份
            title: 标题
            journal: 期刊名
            volume: 卷号
            issue: 期号
            pages: 页码
            
        Returns:
            格式化后的引用字符串
        """
        if not authors:
            author_str = ""
        elif len(authors) > 20:
            # APA格式：超过20个作者时使用省略号
            author_str = ", ".join([self._format_surname_with_initials(author) for author in authors[:19]]) + ", ... " + self._format_surname_with_initials(authors[-1])
        elif len(authors) > 1:
            author_str = ", ".join([self._format_surname_with_initials(author) for author in authors[:-1]]) + ", & " + self._format_surname_with_initials(authors[-1])
        else:
            author_str = self._format_surname_with_initials(authors[0])
        
        parts = [author_str, f"({year})."]
        if title:
            parts.append(title)
        if journal:
            journal_part = journal
            if volume:
                journal_part += f", {volume}"
                if issue:
                    journal_part += f"({issue})"
            if pages:
                journal_part += f", {pages}"
            parts.append(journal_part)
        
        return " ".join([part for part in parts if part])
    
    def _format_mla(
        self, 
        authors: List[str], 
        year: str, 
        title: Optional[str] = None,
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        pages: Optional[str] = None
    ) -> str:
        """
        格式化为MLA格式
        
        Args:
            authors: 作者列表
            year: 年份
            title: 标题
            journal: 期刊名
            volume: 卷号
            pages: 页码
            
        Returns:
            格式化后的引用字符串
        """
        if not authors:
            author_str = ""
        elif len(authors) > 1:
            author_str = self._format_surname_with_initials(authors[0]) + ", et al."
        else:
            author_str = self._format_surname_with_initials(authors[0])
        
        parts = [author_str, title, journal, volume, pages, year]
        return " ".join([part for part in parts if part]) + "."
    
    def _format_chicago(
        self, 
        authors: List[str], 
        year: str, 
        title: Optional[str] = None,
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None
    ) -> str:
        """
        格式化为芝加哥格式
        
        Args:
            authors: 作者列表
            year: 年份
            title: 标题
            journal: 期刊名
            volume: 卷号
            issue: 期号
            pages: 页码
            
        Returns:
            格式化后的引用字符串
        """
        if not authors:
            author_str = ""
        elif len(authors) > 1:
            author_str = self._format_surname_with_initials(authors[0]) + ", et al."
        else:
            author_str = self._format_surname_with_initials(authors[0])
        
        parts = [author_str, f"{year}."]
        if title:
            parts.append(f'"{title}."')
        if journal:
            journal_part = f"<i>{journal}</i>"
            if volume:
                journal_part += f" {volume}"
                if issue:
                    journal_part += f", no. {issue}"
            if pages:
                journal_part += f": {pages}"
            parts.append(journal_part)
        
        return " ".join([part for part in parts if part])
    
    def _format_numeric(self, authors: List[str], year: str) -> str:
        """
        格式化为数字格式
        
        Args:
            authors: 作者列表
            year: 年份
            
        Returns:
            格式化后的引用字符串
        """
        if not authors:
            return f"[{year}]"
        
        # 数字格式通常只显示第一作者+et al.+年份
        first_author = authors[0] if authors else ""
        surname = self._extract_surname(first_author)
        
        return f"[{surname} et al., {year}]"
    
    def _format_surname_with_initials(self, author: str) -> str:
        """
        格式化作者名为姓, 名缩写格式
        
        Args:
            author: 作者名
            
        Returns:
            格式化后的作者名
        """
        if ', ' in author:
            # 已经是 "姓, 名" 格式
            return author
        else:
            # 是 "名 姓" 格式，需要转换
            parts = author.split()
            if len(parts) >= 2:
                surname = parts[-1]
                initials = [name[0] + '.' for name in parts[:-1]]
                return f"{surname}, {' '.join(initials)}"
            else:
                return author
    
    def _extract_surname(self, author: str) -> str:
        """
        从作者名中提取姓氏
        
        Args:
            author: 作者名
            
        Returns:
            姓氏
        """
        # 去除可能的前后空格
        author = author.strip()

        # 如果包含逗号，则逗号前的是姓 (如 "Ohanian, R.")
        if ',' in author:
            return author.split(',')[0].strip()

        # 处理英文名字，提取姓
        parts = author.split()
        if len(parts) == 0:
            return author
        elif len(parts) == 1:
            return parts[0]
        else:
            # 对于英文名字，通常最后一个部分是姓
            return parts[-1]
    
    def _contains_chinese(self, text: str) -> bool:
        """
        检查文本是否包含中文字符
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否包含中文字符
        """
        import re
        return bool(re.search('[\u4e00-\u9fff]', text))