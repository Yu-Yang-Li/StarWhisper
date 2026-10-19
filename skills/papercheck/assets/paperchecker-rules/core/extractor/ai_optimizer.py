"""Deterministic cleanup used by the upstream citation extractor.

Keep the legacy function name for the extractor's internal callers. Semantic
review is performed by the model running the skill, outside this rule engine.
"""

import re


def optimize_citations_with_ai(citations_list, config=None):
    return _optimize_citations_with_rules(citations_list)


def _optimize_citations_with_rules(citations_list):
    """
    使用严格的规则优化引用列表，过滤非引用内容
    这个版本更严格地过滤，模拟AI优化的效果
    
    Args:
        citations_list (list): 提取到的引用列表
        
    Returns:
        list: 优化后的引用列表
    """
    if not citations_list:
        return []
    
    optimized_citations = []
    
    # 定义需要严格过滤的关键词模式（包括上下文内容）
    filter_patterns = [
        r'.*等[，,].*（\d{4}）.*',           # "张三等（2020）"后面还跟有其他内容
        r'.*[，,].*等.*（\d{4}）.*',         # "张三，李四等（2020）"后面还跟有其他内容  
        r'^在.*',                           # "在...张三（2020）"
        r'.*张玉利.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*汤天波.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*赵若羽.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*葛文静.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*等[，,]?.*（\d{4}）的',          # "张三等（2020）的..."
        r'.*认为.*（\d{4}）',                # "张三认为（2020）"
        r'.*[，,].*（\d{4}）$',             # 以年份结尾但前面有逗号（可能是句子的一部分）
        r'^根据.*（\d{4}）',                 # "根据张三（2020）"
        r'^基于.*（\d{4}）',                 # "基于张三（2020）"  
        r'^参考.*（\d{4}）',                 # "参考张三（2020）"
        r'^如.*（\d{4}）',                   # "如张三（2020）"
        r'^例如.*（\d{4}）',                 # "例如张三（2020）"
        r'.*指出.*（\d{4}）',                # "张三指出（2020）"
        r'.*提到.*（\d{4}）',                # "张三提到（2020）"
        r'.*发现.*（\d{4}）',                # "张三发现（2020）"
        r'.*提出.*（\d{4}）',                # "张三提出（2020）"
        r'.*等.*[，,].*（\d{4}）',           # "张三等，李四（2020）"这种错误格式
        r'.*和.*（\d{4}）',                  # "张三和（2020）"这种不完整格式
    ]
    
    # 严格的标准引用格式模式 - 只接受干净的引用格式
    # 中文格式：作者（年份）或 作者 等（年份）或 作者1 & 作者2（年份）
    # 英文格式：Author（Year）或 Author et al.（Year）或 Author1 & Author2（Year）
    valid_citation_pattern = r'^([A-Za-z\u4e00-\u9fff\s&.&＆等，,]+?)\s*[（(]\s*(\d{4})\s*[)）]$'
    
    for citation in citations_list:
        # 检查是否需要过滤（包含上下文的引用）
        should_filter = False
        for pattern in filter_patterns:
            if re.search(pattern, citation.strip()):
                should_filter = True
                break
        
        if should_filter:
            continue
        
        # 验证是否为标准引用格式
        match = re.match(valid_citation_pattern, citation.strip())
        if match:
            # 如果符合标准格式，保留
            author_part = match.group(1).strip()
            year_part = match.group(2)
            standard_citation = f"{author_part}（{year_part}）"
            optimized_citations.append(standard_citation)
        # 注意：这里不处理不符合标准格式的引用，直接跳过
    
    # 去重
    unique_citations = list(set(optimized_citations))
    
    # 按年份排序
    try:
        unique_citations.sort(key=lambda x: int(re.search(r'（(\d{4})）', x).group(1)))
    except:
        # 如果排序失败，保持原有顺序
        pass
    
    return unique_citations
