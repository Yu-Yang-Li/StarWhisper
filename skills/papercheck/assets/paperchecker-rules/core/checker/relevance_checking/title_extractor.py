"""
文档标题提取工具
从文档内容中提取实际标题，而不是使用文件名
"""
import re
from typing import Optional


def extract_title_from_content(content: list) -> Optional[str]:
    """
    从文档内容中提取标题

    Args:
        content: 文档内容列表

    Returns:
        提取到的标题，如果未找到则返回None
    """
    if not content:
        return None

    # 定义标题的可能模式
    title_patterns = [
        # 包含论文相关关键词的标题
        r'^\s*(.*?)(?:题目|标题|论文题目|研究题目|毕业论文|学位论文|硕士论文|博士论文|课程论文|开题报告|毕业设计|研究|分析|探讨|综述|调查报告|研究综述|实证研究|理论研究|实验研究|应用研究|系统设计|算法研究|模型构建|优化方法|解决方案|研究进展|发展现状|问题及对策|影响因素分析|比较研究|案例分析|实证分析|理论分析|文献综述|技术综述|综述报告|研究方法|研究方案|研究计划|研究背景|研究目的|研究意义|研究内容|研究方法|研究结果|研究结论|摘要|前言|引言|绪论|导论|背景|目的|意义|现状|发展|趋势|问题|对策|策略|方案|设计|实现|应用|效果|评价|分析|讨论|结论|建议|展望|参考文献|致谢|附录).*?$',
        # 中文标题模式（可能包含数字编号）
        r'^\s*[一二三四五六七八九十0-9]{0,2}[、.\s]*([\\u4e00-\\u9fa5a-zA-Z0-9\\s\\-_]+)$',
        # 简单的标题模式（较长的首行，不含句号等标点）
        r'^\s*([\\u4e00-\\u9fa5a-zA-Z0-9\\s\\-_,，：:【】\[\]()（）]{8,100})$',
        # 可能是标题的模式（不含常见段落结尾标点）
        r'^\s*([\\u4e00-\\u9fa5a-zA-Z0-9\\s\\-_,，：:【】\[\]()（）]{8,100})[。！？.!?]*$',
    ]

    # 检查文档的前几行，通常标题会在前面
    for i, line in enumerate(content[:10]):  # 检查前10行
        if line and isinstance(line, str):
            line = line.strip()
            if not line:
                continue

            # 检查是否符合标题模式
            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip() if match.lastindex else line.strip()
                    # 移除可能的标点符号
                    title = re.sub(r'[。！？.!?；;，,]*$', '', title).strip()
                    if 4 <= len(title) <= 100 and len(title) > 0:  # 标题长度合理
                        return title

    # 特别检查前3行，看是否有明显是标题的内容
    for i, line in enumerate(content[:3]):
        if line and isinstance(line, str):
            line = line.strip()
            if not line:
                continue

            # 检查是否是较长的、包含中文或英文的行，但不包含句号等段落结束标点
            if 8 <= len(line) <= 100:
                # 检查是否包含足够的中文或英文字符
                chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', line))
                english_chars = len(re.findall(r'[a-zA-Z]', line))
                total_chars = len(re.sub(r'\s', '', line))

                # 如果中文和英文字符比例较高，且不以常见段落结束标点结尾，则可能是标题
                if (total_chars > 0 and (chinese_chars + english_chars) / total_chars > 0.4 and
                    not line.endswith(('。', '！', '？', '.', '!', '?', '；', ';'))):
                    # 避免将表格内容或列表项误认为标题
                    if not re.match(r'^\s*[0-9]+[.、]', line) and not line.startswith('- ') and not line.startswith('* '):
                        # 避免将作者、单位等信息误认为标题
                        if not any(keyword in line.lower() for keyword in ['作者', '单位', 'email', '通讯', '地址', '邮编', '电话']):
                            return line

    # 如果没有找到明确的标题，返回第一个有意义的段落（但要确保不是其他内容）
    for line in content[:5]:
        if line and isinstance(line, str):
            line = line.strip()
            if 8 <= len(line) <= 100 and line:
                # 检查是否包含足够的中文或英文字符
                chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', line))
                english_chars = len(re.findall(r'[a-zA-Z]', line))
                total_chars = len(re.sub(r'\s', '', line))

                if (total_chars > 0 and (chinese_chars + english_chars) / total_chars > 0.4 and
                    not line.endswith(('。', '！', '？', '.', '!', '?', '；', ';'))):
                    return line

    return None