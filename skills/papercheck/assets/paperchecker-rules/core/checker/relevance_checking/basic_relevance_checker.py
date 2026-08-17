"""
基础相关性检查器 - 当AI服务不可用时的备用方案
使用文本匹配算法进行基本的相关性评估
"""
import re
from difflib import SequenceMatcher
from typing import Dict, Any


def basic_relevance_check(document_title: str, target_content: str) -> Dict[str, Any]:
    """
    基础的相关性检查，使用文本匹配算法
    
    Args:
        document_title: 文档标题
        target_content: 目标内容（用于比较）
    
    Returns:
        包含相关性评估结果的字典
    """
    # 清理文本
    doc_title_clean = re.sub(r'[^\w\s]', '', document_title.lower())
    target_content_clean = re.sub(r'[^\w\s]', '', target_content.lower())
    
    # 计算文本相似度
    similarity_ratio = SequenceMatcher(None, doc_title_clean, target_content_clean).ratio()
    
    # 基于相似度计算相关性评分
    relevance_score = int(similarity_ratio * 10)  # 转换为0-10分制
    
    # 检查关键词匹配
    keywords_matched = 0
    total_keywords = 0
    
    # 从目标内容中提取关键词
    target_words = set(target_content_clean.split())
    title_words = set(doc_title_clean.split())
    
    if target_words:
        keywords_matched = len(target_words.intersection(title_words))
        total_keywords = len(target_words)
        keyword_match_ratio = keywords_matched / total_keywords if total_keywords > 0 else 0
    else:
        keyword_match_ratio = 0
    
    # 综合评分（文本相似度和关键词匹配各占一部分）
    combined_score = int((similarity_ratio * 0.6 + keyword_match_ratio * 0.4) * 10)
    
    # 确保评分在合理范围内
    relevance_score = max(0, min(10, combined_score))
    
    # 判断是否适合引用
    is_suitable = relevance_score > 3
    
    # 生成简要依据
    if relevance_score >= 7:
        basis = f"标题与内容高度相关，匹配度{relevance_score}/10"
    elif relevance_score >= 4:
        basis = f"标题与内容有一定相关性，匹配度{relevance_score}/10"
    else:
        basis = f"标题与内容相关性较低，匹配度{relevance_score}/10"
    
    return {
        "relevance_score": relevance_score,
        "is_suitable_for_citation": is_suitable,
        "brief_basis": basis,
        "detailed_reasoning": f"文本相似度: {similarity_ratio:.2f}, 关键词匹配率: {keyword_match_ratio:.2f}, 匹配关键词数: {keywords_matched}/{total_keywords}"
    }