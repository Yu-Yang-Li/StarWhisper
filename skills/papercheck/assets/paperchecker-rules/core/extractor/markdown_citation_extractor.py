#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown引文提取器
从Markdown内容中提取作者-年份格式的引文
"""

import re
from .citation_optimizer import optimize_citations_with_ai


def extract_references_from_markdown(markdown_content: str, config=None):
    """从Markdown内容中提取作者年份格式引用"""
    patterns = [
        # 中文格式: 作者（年份）
        r'([\u4e00-\u9fa5\w\s\.&＆,，]+?)\s*[（(](\d{4})[）)]',
        # 西文格式: Author (year)
        r'([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s*&\s*[A-Z][a-z]+(?:\s+[A-Z]\.)?)?)\s*[（(](\d{4})[）)]',
        # et al. 格式
        r'([A-Z][a-z]+\s+et\s+al\.)[\s\u00a0]*[（(](\d{4})[）)]',
        # 等 格式
        r'([\u4e00-\u9fa5\w\s]+?)\s+等[\s\u00a0]*[（(](\d{4})[）)]',
        # 多作者格式: Johnson & Brown (2021)
        r'([A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)+)\s*[（(](\d{4})[）)]',
        # 简单的年份格式: (Smith, 2020)
        r'[（(]([A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)?),?\s*(\d{4})[）)]',
    ]

    references = []
    for pattern in patterns:
        matches = list(re.finditer(pattern, markdown_content))
        for match in matches:
            if len(match.groups()) >= 2:
                # 根据正则模式调整组索引
                if pattern == r'[（(]([A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)?),?\s*(\d{4})[）)]':
                    # 这个模式中，作者在group(1)，年份在group(2)
                    author = match.group(1).strip()
                    year = match.group(2).strip()
                else:
                    # 其他模式中，作者在group(1)，年份在group(2)
                    author = match.group(1).strip()
                    year = match.group(2).strip()
                
                if len(author) >= 2 and year.isdigit() and len(year) == 4:
                    reference = f"{author}（{year}）"
                    references.append(reference)

    # 去重
    unique_references = list(dict.fromkeys(references))  # 保持顺序的去重

    # 使用AI优化
    if unique_references and optimize_citations_with_ai:
        try:
            optimized = optimize_citations_with_ai(unique_references, config)
            return optimized
        except Exception as e:
            print(f"AI优化引文时出错: {e}")
            return unique_references
    
    return unique_references


def extract_western_references_from_markdown(markdown_content: str, config=None):
    """从Markdown内容中提取西式格式引用"""
    # 西式引用格式
    western_patterns = [
        # 格式: Lastname, F. (year)
        r'([A-Z][a-z]+,\s*[A-Z]\.)\s*\((\d{4})\)',
        # 格式: Lastname, F. & Lastname, G. (year)
        r'([A-Z][a-z]+,\s*[A-Z]\.(?:\s*&\s*[A-Z][a-z]+,\s*[A-Z]\.)*)\s*\((\d{4})\)',
        # 格式: Lastname (year)
        r'([A-Z][a-z]+)\s*\((\d{4})\)',
        # 格式: Lastname and Lastname (year)
        r'([A-Z][a-z]+\s+and\s+[A-Z][a-z]+)\s*\((\d{4})\)',
        # 格式: (Lastname, year)
        r'\(([A-Z][a-z]+),\s*(\d{4})\)',
    ]

    western_refs = []
    for pattern in western_patterns:
        matches = list(re.finditer(pattern, markdown_content))
        for match in matches:
            if len(match.groups()) >= 2:
                author = match.group(1).strip()
                year = match.group(2).strip()
                
                if len(author) >= 2 and year.isdigit() and len(year) == 4:
                    reference = f"{author}（{year}）"
                    western_refs.append(reference)

    # 去重
    unique_references = list(dict.fromkeys(western_refs))  # 保持顺序的去重

    # 使用AI优化
    if unique_references and optimize_citations_with_ai:
        try:
            optimized = optimize_citations_with_ai(unique_references, config)
            return optimized
        except Exception as e:
            print(f"AI优化西式引文时出错: {e}")
            return unique_references
    
    return unique_references


def extract_citations_from_markdown(markdown_content: str, config=None):
    """从Markdown内容中提取所有类型的引文"""
    # 提取中文和西文引用
    chinese_citations = extract_references_from_markdown(markdown_content, config)
    western_citations = extract_western_references_from_markdown(markdown_content, config)
    
    # 合并并去重
    all_citations = chinese_citations + western_citations
    unique_citations = list(dict.fromkeys(all_citations))  # 保持顺序去重
    
    return unique_citations


if __name__ == "__main__":
    # 测试代码
    sample_markdown = """
    这是一篇关于共享经济的研究论文。近年来，共享经济模式得到了广泛关注 (Botsman & Rogers, 2011)。
    Felson (1978) 最早提出了协同消费的概念。随后，Schor (2014) 对共享经济进行了深入探讨。
    在中国，汤天波（2015）等学者也对这一模式进行了研究。汪寿阳等（2015）进一步分析了商业模式的创新。
    罗珉 等（2015）认为互联网时代的商业模式创新值得关注。Johnson & Christensen（2008）提出了业务模式创新的理论框架。
    """
    
    citations = extract_citations_from_markdown(sample_markdown)
    print("提取到的引文:")
    for citation in citations:
        print(f"  - {citation}")