"""
引用分析器
负责分析文档中的引用和参考文献匹配情况
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_dir)

from core.checker.citation_checker import CitationChecker
from core.checker.citation_checking.reference_mapper import map_author_year_citation_to_reference
from core.extractor.citation_extraction.citation_extractor import extract_authors_from_reference
from core.extractor.citation_extraction.citation_formatter import CitationFormatter
from config.citation_format_config import CitationFormatConfig, get_default_citation_format_config


def analyze_document(doc_path: str) -> Dict[str, Any]:
    """分析文档并返回引用分析报告"""

    print(f"=== 开始分析文档: {doc_path} ===")

    # 使用真实文档 - 首先尝试当前目录的文档
    print("创建CitationChecker实例...")
    # 尝试使用完整路径到配置文件
    # 直接使用config.json配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "config.json")
    checker = CitationChecker(doc_path, config_path)

    try:
        # 提取文档中的引用和参考文献
        print("开始提取引用和参考文献...")
        checker.extract_citations_and_references()

        print(f"提取到 {len(checker.citations)} 个引用和 {len(checker.references)} 个参考文献条目")

        # 统计匹配结果
        matched_count = 0
        unmatched_count = 0
        corrected_count = 0
        formatted_count = 0
        corrections_needed = []
        formatting_needed = []

        print("\n=== 映射结果 ===")

        # 准备详细报告数据
        detailed_report = {
            "test_date": datetime.now().isoformat(),
            "document": doc_path,
            "total_citations": len(checker.citations),
            "total_references": len(checker.references),
            "results": []
        }

        # 存储所有映射结果
        all_mapping_results = []

        # 测试所有引用 (作者年份格式和数字格式)
        test_citations = checker.citations

        for i, citation in enumerate(test_citations, 1):
            # 检查是否是[AUTH:]格式的引用
            if citation.startswith("[AUTH:"):
                actual_citation = citation[6:-1]  # 去掉[AUTH:]前缀和后缀
            else:
                # 数字格式的引用，直接使用
                actual_citation = citation

            # 执行映射 - 区分数字引用和作者年份引用
            if citation.startswith("[AUTH:]"):
                result = map_author_year_citation_to_reference(actual_citation, checker.references)
            else:
                # 对数字引用执行其他处理（如果需要）
                result = None
                # 查找对应的数字参考文献
                import re
                for ref in checker.references:
                    ref_match = re.search(r'^\[\d+\]', ref['text'])
                    if ref_match and ref_match.group() == citation:
                        result = {
                            'original_citation': citation,
                            'corrected_citation': citation,
                            'reference': ref
                        }
                        break

            # 如果匹配成功，提取作者信息
            if result and citation.startswith("[AUTH:]"):
                reference_text = result['reference']['text']
                authors, has_et_al = extract_authors_from_reference(reference_text)
                result['authors'] = authors
                result['has_et_al'] = has_et_al

                # 获取引用格式配置和格式化器
                citation_config = get_default_citation_format_config()
                formatter = CitationFormatter(citation_config)

                # 根据作者信息格式化引用
                year = result['reference'].get('year', '')
                if not year and 'corrected_citation' in result:
                    # 尝试从修正后的引用中提取年份
                    import re
                    year_match = re.search(r'（(\d{4})）', result['corrected_citation'])
                    if year_match:
                        year = year_match.group(1)

                if year and authors:
                    formatted_citation = formatter.format_citation(authors, year)
                    result['formatted_citation'] = formatted_citation
                else:
                    result['formatted_citation'] = result['corrected_citation']

            # 存储映射结果
            if result:
                all_mapping_results.append(result)

            print(f"\n{i}. 测试引用: {actual_citation}")

            # 记录结果到详细报告
            report_entry = {
                "citation_index": i,
                "original_citation": actual_citation,
                "matched": result is not None,
                "needs_correction": False,
                "needs_formatting": False
            }

            if result:
                matched_count += 1
                print("  匹配成功!")
                print(f"  匹配的参考文献: {result['reference']['text'][:100]}...")

                # 打印作者信息
                if 'authors' in result and result['authors']:
                    print(f"  提取的作者: {', '.join(result['authors'])}")
                    if result.get('has_et_al', False):
                        print(f"  有et al.标记: 是")
                else:
                    print("  未能提取作者信息")

                report_entry.update({
                    "reference_text": result['reference']['text'],
                    "reference_year": result['reference'].get('year', '未知'),
                    "matched_year": result['reference'].get('year', '未知'),
                    "authors": result.get('authors', []),
                    "has_et_al": result.get('has_et_al', False)
                })

                # 检查是否需要修正
                needs_correction = result['corrected_citation'] != result['original_citation']
                needs_formatting = result.get('formatted_citation', result['corrected_citation']) != result['corrected_citation']

                if needs_correction or needs_formatting:
                    if needs_correction:
                        corrected_count += 1
                        print(f"  修正后的引用: {result['corrected_citation']}")
                        report_entry.update({
                            "corrected_citation": result['corrected_citation'],
                            "needs_correction": True,
                            "correction_reason": "年份不一致"
                        })
                        corrections_needed.append({
                            "original": result['original_citation'],
                            "corrected": result['corrected_citation'],
                            "reference": result['reference']['text'][:100] + "..."
                        })

                    if needs_formatting:
                        formatted_count += 1
                        print(f"  格式化后的引用: {result['formatted_citation']}")
                        report_entry.update({
                            "formatted_citation": result['formatted_citation'],
                            "needs_formatting": True,
                            "formatting_reason": "作者格式不一致"
                        })
                        formatting_needed.append({
                            "original": result['corrected_citation'] if needs_correction else result['original_citation'],
                            "formatted": result['formatted_citation'],
                            "reference": result['reference']['text'][:100] + "..."
                        })

                    print(f"  原始引用: {result['original_citation']}")
                else:
                    print(f"  引用无须修正: {result['corrected_citation']}")
                    report_entry["corrected_citation"] = result['corrected_citation']
            else:
                unmatched_count += 1
                print("  未找到匹配的参考文献")

            detailed_report["results"].append(report_entry)

        # 识别未被引用的参考文献
        used_references = {result['reference']['text'] for result in all_mapping_results if result}
        unused_references = [ref for ref in checker.references if ref['text'] not in used_references]

        # 添加统计信息到报告
        detailed_report.update({
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "corrected_count": corrected_count,
            "formatted_count": formatted_count,
            "unused_references_count": len(unused_references),
            "match_rate": f"{matched_count/len(test_citations)*100:.1f}%" if len(test_citations) > 0 else "0.0%",
            "corrections_needed": corrections_needed,
            "formatting_needed": formatting_needed,
            "unused_references": unused_references
        })

        print(f"\n=== 测试总结 ===")
        print(f"测试引用数量: {len(test_citations)}")
        print(f"成功匹配: {matched_count}")
        print(f"未匹配: {unmatched_count}")
        print(f"需要修正: {corrected_count}")
        print(f"未使用参考文献: {len(unused_references)}")
        print(f"需要格式化: {formatted_count}")
        print(f"匹配率: {matched_count/len(test_citations)*100:.1f}%" if len(test_citations) > 0 else "0.0%")

        # 打印需要修正的引用
        if corrections_needed:
            print(f"\n需要修正的引用 ({corrected_count} 个):")
            for correction in corrections_needed:
                print(f"  - 原始: {correction['original']}")
                print(f"    修正: {correction['corrected']}")
                print(f"    参考文献: {correction['reference']}")

        # 打印需要格式化的引用
        if formatting_needed:
            print(f"\n需要格式化的引用 ({formatted_count} 个):")
            for formatting in formatting_needed:
                print(f"  - 原始: {formatting['original']}")
                print(f"    格式化: {formatting['formatted']}")
                print(f"    参考文献: {formatting['reference']}")
                print("--------------------------------------------------")

        if unused_references:
            print(f"\n未被引用的参考文献 ({len(unused_references)} 个):")
            for i, ref in enumerate(unused_references, 1):
                print(f"  {i}. {ref['text']}")
                print("--------------------------------------------------")

        return detailed_report
    finally:
        # 清理资源
        if hasattr(checker, 'cleanup') and callable(getattr(checker, 'cleanup')):
            checker.cleanup()