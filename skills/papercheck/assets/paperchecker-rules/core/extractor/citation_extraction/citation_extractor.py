"""
引用提取器
负责从参考文献中提取作者、年份等信息
"""

import re
from typing import Dict, Any, List


def contains_chinese(text):
    """检查文本是否包含中文字符"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def extract_authors_from_reference(reference_text):
    """
    从参考文献条目中提取作者列表

    参数:
        reference_text: 参考文献文本

    返回:
        作者列表, 是否有et al.标记
    """
    # 首先检查是否是中文文献
    if contains_chinese(reference_text[:50]) or any(char in reference_text for char in ['，', '。', '期刊', '杂志']):
        authors = extract_chinese_authors(reference_text)
        return authors, False
    else:
        return extract_english_authors(reference_text)

def extract_chinese_authors(reference_text):
    """
    从中文参考文献条目中提取作者列表
    """
    authors = []

    # 中文文献模式: 作者1,作者2.文章名...
    chinese_pattern = r'^([^\.]+?)\.'  # 匹配第一个句号之前的内容

    match = re.search(chinese_pattern, reference_text)
    if match:
        authors_str = match.group(1)
        # 分割作者 (中文使用逗号或顿号分隔)
        if '，' in authors_str:
            authors = [author.strip() for author in authors_str.split('，') if author.strip()]
        elif ',' in authors_str:
            authors = [author.strip() for author in authors_str.split(',') if author.strip()]
        else:
            authors = [authors_str.strip()]

        # 处理"等"字
        authors = [author for author in authors if author != '等']

    return authors

def extract_english_authors(reference_text):
    """
    从英文参考文献条目中提取作者列表

    返回:
        作者列表, 是否有et al.标记
    """
    authors = []
    has_et_al = False

    # 检查是否有et al.
    if re.search(r'et al\.?', reference_text, re.IGNORECASE):
        has_et_al = True

    # 提取作者部分 (通常在第一个句号或括号之前)
    match = re.search(r'^([^\(]+?)\(\d{4}\)', reference_text)
    if match:
        authors_str = match.group(1)
    else:
        match = re.search(r'^([^\.]+?)\.\s*\(\d{4}\)', reference_text)
        if match:
            authors_str = match.group(1)
        else:
            match = re.search(r'^([^\.]+?)\.', reference_text)
            if match:
                authors_str = match.group(1)
            else:
                # 如果以上模式都不匹配，尝试提取第一个逗号或句号之前的内容
                first_comma = reference_text.find(',')
                first_period = reference_text.find('.')

                if first_comma != -1 and (first_period == -1 or first_comma < first_period):
                    authors_str = reference_text[:first_comma]
                elif first_period != -1:
                    authors_str = reference_text[:first_period]
                else:
                    return authors, has_et_al

    # 处理"and"和"&"连接符
    authors_str = re.sub(r'\s+and\s+', ', ', authors_str)
    authors_str = re.sub(r'\s*&\s*', ', ', authors_str)

    # 移除"et al."等缩写
    authors_str = re.sub(r',?\s*et al\.?', '', authors_str, flags=re.IGNORECASE)

    # 改进的作者分割逻辑
    # 使用更智能的方法分割作者，考虑名字缩写中的逗号
    author_list = []
    current_author = ""
    in_abbreviation = False

    for char in authors_str:
        if char == ',' and not in_abbreviation:
            if current_author.strip():
                author_list.append(current_author.strip())
            current_author = ""
        else:
            current_author += char
            if char == '.':
                in_abbreviation = True
            elif char == ' ' and in_abbreviation:
                in_abbreviation = False

    if current_author.strip():
        author_list.append(current_author.strip())

    # 处理每个作者
    for author in author_list:
        # 处理名字缩写 (如 "A." 或 "A. B.")
        if re.match(r'^[A-Z]\.(?:\s*[A-Z]\.)?$', author):
            if authors:
                # 将缩写合并到前一个作者
                authors[-1] = authors[-1] + ' ' + author
            else:
                # 第一个就是缩写，可能是单字母作者名
                authors.append(author)
        else:
            authors.append(author)

    return authors, has_et_al

def extract_surname(author):
    """
    从作者字符串中提取姓氏

    参数:
        author: 作者字符串

    返回:
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
        # 对于英文名字，通常第一个部分是姓
        return parts[0]

def format_citation_by_authors(authors, year, original_citation, has_et_al=False):
    """
    根据作者数量和类型格式化引用

    参数:
        authors: 作者列表
        year: 年份
        original_citation: 原始引用
        has_et_al: 是否有et al.标记

    返回:
        格式化后的引用
    """
    if not authors:
        return original_citation

    # 判断是否是中文作者
    is_chinese = contains_chinese(authors[0]) if authors else False

    if is_chinese:
        # 中文作者
        if len(authors) > 1 or has_et_al:
            return f"{authors[0]} 等（{year}）"
        else:
            return f"{authors[0]}（{year}）"
    else:
        # 欧美作者
        # 提取作者的姓
        surnames = [extract_surname(author) for author in authors]

        # 处理机构或团体作者
        if len(surnames) == 1 and (len(surnames[0].split()) > 2 or surnames[0].isupper()):
            return f"{surnames[0]}（{year}）"

        # 如果有et al.标记，或者作者数量大于2，使用et al.
        if has_et_al or len(surnames) > 2:
            return f"{surnames[0]} et al.（{year}）"
        elif len(surnames) == 2:
            return f"{surnames[0]} & {surnames[1]}（{year}）"
        else:
            return f"{surnames[0]}（{year}）"