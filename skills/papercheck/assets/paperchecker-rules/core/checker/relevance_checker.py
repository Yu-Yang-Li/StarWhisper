"""
相关性检查器 - 检查文献引用的相关性
支持快速检测（基于标题）和准确检查（基于全文）
"""
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from .base_checker import BaseChecker
from models.document import Document
from models.compliance import ComplianceResult, CheckType
from core.ai.ai_client import ai_generate


class RelevanceChecker(BaseChecker):
    """相关性检查器 - 检查文献引用的相关性"""

    def __init__(self, use_full_content: bool = False):
        """
        初始化相关性检查器

        Args:
            use_full_content: 是否使用全文内容进行检查（False为快速检测，True为准确检查）
        """
        self.use_full_content = use_full_content
        # 设置提示词模板路径
        self.prompts_dir = Path(__file__).parent / "relevance_checking" / "prompts"

    def get_check_type(self) -> CheckType:
        return CheckType.RELEVANCE

    def get_check_name(self) -> str:
        return "relevance_checker"

    def check(self, document: Document, target_content: str, task_type: str = "文章整体") -> ComplianceResult:
        """
        执行相关性检查

        Args:
            document: 要检查的文献文档对象
            target_content: 对比内容（文章标题或段落内容）
            task_type: 任务类型，"文章整体" 或 "段落"

        Returns:
            ComplianceResult: 检查结果
        """
        if self.use_full_content:
            return self._accurate_check(document, target_content, task_type)
        else:
            return self._quick_check(document, target_content, task_type)

    def _quick_check(self, document: Document, target_content: str, task_type: str) -> ComplianceResult:
        """
        快速检测（基于标题）

        Args:
            document: 要检查的文献文档对象
            target_content: 对比内容（文章标题或段落内容）
            task_type: 任务类型，"文章整体" 或 "段落"

        Returns:
            ComplianceResult: 检查结果
        """
        # 读取提示词模板
        prompt_template = (self.prompts_dir / "quick_check_prompt.txt").read_text(encoding='utf-8')

        # 替换模板中的占位符
        # 优先使用文档元数据中的标题，如果不存在则尝试从文档内容中提取
        document_title = document.metadata.get('title', None) if hasattr(document, 'metadata') else None

        if not document_title or document_title == '未知标题':
            # 如果文档元数据中没有标题，尝试从文档内容中提取
            if hasattr(document, 'content') and document.content:
                from .relevance_checking.title_extractor import extract_title_from_content
                extracted_title = extract_title_from_content(document.content)
                if extracted_title:
                    document_title = extracted_title
                else:
                    # 如果仍然无法提取标题，使用文件名作为后备
                    if hasattr(document, 'metadata') and 'file_path' in document.metadata:
                        import os  # 确保os模块可用
                        document_title = os.path.splitext(os.path.basename(document.metadata['file_path']))[0]
                    else:
                        document_title = '未知标题'
        # else: document_title 已经有值，不需要修改

        # 确保document_title是字符串类型并且不包含可能导致format错误的字符
        if not isinstance(document_title, str):
            document_title = str(document_title) if document_title is not None else '未知标题'

        # 防止document_title包含特殊字符导致format错误
        # 如果document_title包含换行符或其他特殊字符，进行清理
        document_title = str(document_title).replace('\n', ' ').replace('\r', ' ').replace('"', "'").replace('{', '(').replace('}', ')').strip()

        try:
            prompt = prompt_template.format(task_type=task_type, document_title=document_title, target_content=target_content)
        except KeyError as e:
            # 如果格式化失败，使用安全的备选方案
            print(f"格式化提示词模板时出错: {e}")
            # 使用更安全的字符串替换方法
            prompt = prompt_template.replace("{task_type}", str(task_type)) \
                                   .replace("{document_title}", document_title) \
                                   .replace("{target_content}", str(target_content))

        # 调用AI生成结果
        ai_response = ai_generate(prompt)

        # 检查AI服务是否可用
        if "错误：" in ai_response or "AI服务未配置有效密钥" in ai_response:
            # 如果AI服务不可用，使用基础检查方法
            from .relevance_checking.basic_relevance_checker import basic_relevance_check
            result = basic_relevance_check(document_title, target_content)
            result["task_type"] = task_type  # 添加task_type字段
        else:
            # 确保 ai_response 不是 None
            if ai_response is None:
                ai_response = "AI服务返回空响应"

            # 解析AI响应
            result = self._parse_ai_response(ai_response, task_type)

        # 创建合规性结果对象
        compliance_result = ComplianceResult(
            check_type=CheckType.RELEVANCE,
            is_compliant=result['is_suitable_for_citation'],
            issues=[result],
            statistics={
                "relevance_score": result['relevance_score'],
                "task_type": task_type,
                "check_method": "quick_check"
            },
            metadata={
                "ai_response": ai_response,
                "checker_version": "1.0.0"
            }
        )

        return compliance_result

    def _accurate_check(self, document: Document, target_content: str, task_type: str) -> ComplianceResult:
        """
        准确检查（基于全文）

        Args:
            document: 要检查的文献文档对象
            target_content: 对比内容（文章标题或段落内容）
            task_type: 任务类型，"文章整体" 或 "段落"

        Returns:
            ComplianceResult: 检查结果
        """
        # 获取文献的完整内容
        full_content = self._get_full_content(document)

        # 确保 full_content 不是 None
        if full_content is None:
            full_content = "无法获取文献内容"

        # 读取提示词模板
        prompt_template = (self.prompts_dir / "accurate_check_prompt.txt").read_text(encoding='utf-8')
        
        # 替换模板中的占位符
        prompt = prompt_template.format(task_type=task_type, full_content=full_content, target_content=target_content)

        # 调用AI生成结果
        ai_response = ai_generate(prompt)

        # 检查AI服务是否可用
        if "错误：" in ai_response or "AI服务未配置有效密钥" in ai_response:
            # 如果AI服务不可用，使用基础检查方法
            # 使用文档标题和目标内容进行基础检查
            document_title = document.metadata.get('title', '未知标题') if hasattr(document, 'metadata') else '未知标题'
            from .relevance_checking.basic_relevance_checker import basic_relevance_check
            result = basic_relevance_check(document_title, target_content)
            result["task_type"] = task_type  # 添加task_type字段
        else:
            # 确保 ai_response 不是 None
            if ai_response is None:
                ai_response = "AI服务返回空响应"

            # 解析AI响应
            result = self._parse_ai_response(ai_response, task_type, detailed=True)

        # 创建合规性结果对象
        compliance_result = ComplianceResult(
            check_type=CheckType.RELEVANCE,
            is_compliant=result['is_suitable_for_citation'],
            issues=[result],
            statistics={
                "relevance_score": result['relevance_score'],
                "task_type": task_type,
                "check_method": "accurate_check"
            },
            metadata={
                "ai_response": ai_response,
                "checker_version": "1.0.0"
            }
        )

        return compliance_result

    def _get_full_content(self, document: Document) -> str:
        """
        获取文献的完整内容

        Args:
            document: 文献文档对象

        Returns:
            str: 文献的完整内容
        """
        # 如果文档已经有完整内容，则直接返回
        if hasattr(document, 'full_text') and document.full_text:
            return document.full_text

        # 检查文档的metadata中是否有文件路径
        file_path = None
        if hasattr(document, 'metadata') and 'file_path' in document.metadata:
            file_path = document.metadata['file_path']
        elif hasattr(document, 'file_path'):
            file_path = document.file_path

        # 如果有文件路径，使用适当的提取器提取内容
        import os  # 确保os模块可用
        if file_path and os.path.exists(file_path):
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')

            if ext in ['doc', 'docx']:
                from core.extractor.word_extractor import WordExtractor
                extractor = WordExtractor()
            elif ext == 'pdf':
                # 使用PDF提取器
                from core.extractor.pdf_extractor import PDFExtractor
                extractor = PDFExtractor()
            else:
                # 如果文件类型不支持，尝试获取文档标题用于AI搜索
                document_title = document.metadata.get('title', '') if hasattr(document, 'metadata') else ''
                return document_title if document_title is not None else "无法获取文献内容"

            try:
                # 提取文档内容
                extracted_document = extractor.extract(file_path)
                # 将提取的段落合并为完整内容
                if extracted_document.content and extracted_document.content is not None:
                    # 过滤掉 None 值并确保所有元素都是字符串
                    filtered_content = [str(item) if item is not None else "" for item in extracted_document.content]
                    full_content = '\n'.join(filtered_content)
                else:
                    full_content = "文档内容为空"
                return full_content
            except Exception as e:
                print(f"提取文档内容失败: {e}")
                # 如果提取失败，尝试获取文档标题用于AI搜索
                document_title = document.metadata.get('title', '') if hasattr(document, 'metadata') else ''
                return document_title if document_title is not None else f"提取文档内容失败: {str(e)}"

        # 检查文档的metadata中是否有PDF URL
        pdf_url = None
        if hasattr(document, 'metadata') and 'pdf_url' in document.metadata:
            pdf_url = document.metadata['pdf_url']

        # 如果文档有PDF URL，则下载并转换PDF
        if pdf_url:
            pdf_content = self._download_and_convert_pdf(pdf_url)
            if pdf_content and "无法" not in pdf_content:
                return pdf_content

        # 获取文档标题用于AI搜索
        document_title = document.metadata.get('title', '') if hasattr(document, 'metadata') else ''

        # 如果没有PDF URL或下载失败，尝试通过AI搜索获取文献摘要或内容
        if document_title:
            ai_search_content = self._get_content_from_ai_search(document_title)
            if ai_search_content and "无法" not in ai_search_content:
                return ai_search_content

        # 如果以上方法都失败，返回已有的内容或标题
        return document_title if document_title is not None else "无法获取文献内容"

    def _get_content_from_ai_search(self, title: str) -> str:
        """
        通过AI搜索获取文献内容或摘要

        Args:
            title: 文献标题

        Returns:
            str: 获取到的文献内容或摘要
        """
        try:
            # 读取提示词模板
            prompt_template = (self.prompts_dir / "ai_search_prompt.txt").read_text(encoding='utf-8')

            # 替换模板中的占位符
            prompt = prompt_template.format(title=title)

            # 调用AI生成内容
            ai_response = ai_generate(prompt)

            # 尝试解析AI返回的JSON格式内容
            import json
            import re

            # 提取可能的JSON部分（去除可能的额外文本）
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    json_data = json.loads(json_str)

                    # 构建返回内容，包含摘要、研究方法和结论
                    summary = json_data.get("summary", "")
                    research_method = json_data.get("research_method", "")
                    conclusion = json_data.get("conclusion", "")

                    # 如果摘要中包含"未找到相关文献信息"，则返回相应信息
                    if "未找到相关文献信息" in summary:
                        return f"AI搜索未找到关于'{title}'的相关信息"

                    # 组合信息为完整内容
                    content_parts = []
                    if summary:
                        content_parts.append(f"摘要: {summary}")
                    if research_method:
                        content_parts.append(f"研究方法: {research_method}")
                    if conclusion:
                        content_parts.append(f"结论: {conclusion}")

                    return "\n".join(content_parts)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回原始响应
                    if ai_response and "未找到相关文献信息" not in ai_response:
                        return ai_response
                    else:
                        return f"AI搜索未找到关于'{title}'的相关信息"
            else:
                # 如果没有找到JSON格式，返回原始响应
                if ai_response and "未找到相关文献信息" not in ai_response:
                    return ai_response
                else:
                    return f"AI搜索未找到关于'{title}'的相关信息"
        except Exception as e:
            return f"通过AI搜索获取文献内容时出错: {str(e)}"

    def _download_and_convert_pdf(self, pdf_url: str) -> str:
        """
        下载PDF并转换为文本内容

        Args:
            pdf_url: PDF的URL

        Returns:
            str: 转换后的文本内容
        """
        import os  # 确保在此方法中导入os模块
        try:
            # 下载PDF文件
            response = requests.get(pdf_url)
            if response.status_code == 200:
                # 保存临时PDF文件
                temp_pdf_path = "/tmp/temp_document.pdf"
                with open(temp_pdf_path, 'wb') as f:
                    f.write(response.content)

                # 使用mineru转换PDF为markdown
                from utils.mineru_pdf_converter import convert_pdf_to_markdown
                markdown_content = convert_pdf_to_markdown(temp_pdf_path)

                # 删除临时文件
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

                return markdown_content
            else:
                return f"无法下载PDF文件，状态码: {response.status_code}"
        except Exception as e:
            # 确保即使在异常情况下也清理临时文件
            try:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
            except:
                pass  # 忽略清理临时文件时的错误
            return f"获取PDF内容时出错: {str(e)}"

    def _get_content_from_giisp(self, title: str) -> str:
        """
        通过AI搜索根据标题获取文献内容（GIISP未实现时的替代方案）

        Args:
            title: 文献标题

        Returns:
            str: 获取到的文献内容
        """
        # 直接使用AI搜索功能
        return self._get_content_from_ai_search(title)

    def _parse_ai_response(self, ai_response: str, task_type: str, detailed: bool = False) -> Dict[str, Any]:
        """
        解析AI响应

        Args:
            ai_response: AI的响应文本
            task_type: 任务类型
            detailed: 是否为详细分析结果

        Returns:
            Dict[str, Any]: 解析后的结果
        """
        result = {
            "task_type": task_type,
            "relevance_score": 0,
            "is_suitable_for_citation": False,
            "brief_basis": "",
            "detailed_reasoning": ""
        }

        # 检查ai_response是否为None或非字符串类型
        if ai_response is None:
            result["brief_basis"] = "AI响应为空，无法进行相关性分析"
            return result

        # 确保ai_response是字符串类型
        if not isinstance(ai_response, str):
            result["brief_basis"] = f"AI响应类型错误，期望字符串，实际为{type(ai_response)}，无法进行相关性分析"
            return result

        # 检查AI响应是否包含错误信息
        if "AI服务返回空响应" in ai_response or "错误：" in ai_response or "AI调用错误" in ai_response:
            result["brief_basis"] = f"AI服务调用失败: {ai_response}"
            return result

        # 检查AI响应是否为HTML页面（常见错误响应）
        if "<!DOCTYPE html>" in ai_response or "<html" in ai_response or "<head" in ai_response:
            result["brief_basis"] = "AI服务返回了HTML页面而非文本响应，请检查API配置"
            return result

        # 尝试解析JSON格式的响应（新添加）
        import json
        import re

        # 预处理AI响应，移除可能的多余空白字符
        processed_response = ai_response.strip()

        # 提取可能的JSON部分（去除可能的额外文本）
        # 使用更精确的正则表达式来匹配完整的JSON对象
        # 首先尝试匹配最外层的完整JSON对象
        try:
            brace_count = 0
            start_pos = -1
            json_candidate = ""

            # 找到第一个左花括号
            for i, char in enumerate(processed_response):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        # 找到匹配的闭合括号，提取JSON字符串
                        json_candidate = processed_response[start_pos:i+1]
                        try:
                            json_data = json.loads(json_candidate)

                            # 提取JSON中的字段
                            if 'relevance_score' in json_data:
                                result['relevance_score'] = int(json_data['relevance_score'])
                            if 'is_suitable_for_citation' in json_data:
                                result['is_suitable_for_citation'] = bool(json_data['is_suitable_for_citation'])
                            if 'brief_basis' in json_data:
                                result['brief_basis'] = str(json_data['brief_basis'])
                            if 'reasoning' in json_data:
                                result['detailed_reasoning'] = str(json_data['reasoning'])
                            # 兼容不同的字段名
                            if 'detailed_reasoning' in json_data:
                                result['detailed_reasoning'] = str(json_data['detailed_reasoning'])

                            # 如果JSON解析成功，直接返回结果
                            return result
                        except json.JSONDecodeError:
                            # 如果当前JSON解析失败，继续寻找下一个可能的JSON对象
                            continue
                        except Exception as e:
                            # 记录异常但继续寻找其他可能的JSON对象
                            print(f"解析JSON时发生异常: {e}")
                            continue
        except Exception as e:
            # 如果遍历字符串时出错，记录错误并继续使用其他方法
            print(f"遍历AI响应字符串时发生异常: {e}")

        # 如果上述方法失败，再尝试使用正则表达式提取JSON
        # Python的re模块不支持递归，所以我们使用简单的非贪婪匹配
        try:
            json_match = re.search(r'\{[^{}]*\}', processed_response) or re.search(r'\{.*?\}', processed_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    json_data = json.loads(json_str)

                    # 提取JSON中的字段
                    if 'relevance_score' in json_data:
                        result['relevance_score'] = int(json_data['relevance_score'])
                    if 'is_suitable_for_citation' in json_data:
                        result['is_suitable_for_citation'] = bool(json_data['is_suitable_for_citation'])
                    if 'brief_basis' in json_data:
                        result['brief_basis'] = str(json_data['brief_basis'])
                    if 'reasoning' in json_data:
                        result['detailed_reasoning'] = str(json_data['reasoning'])
                    # 兼容不同的字段名
                    if 'detailed_reasoning' in json_data:
                        result['detailed_reasoning'] = str(json_data['detailed_reasoning'])

                    # 如果JSON解析成功，直接返回结果
                    return result
                except json.JSONDecodeError:
                    # JSON解析失败，继续使用原来的解析方法
                    pass
                except Exception as e:
                    # 其他异常也继续使用原来的解析方法
                    print(f"解析JSON时发生异常: {e}")
                    pass
        except Exception as e:
            # 如果正则表达式匹配失败，记录错误并继续使用原来的解析方法
            print(f"正则表达式匹配JSON时发生异常: {e}")
            pass

        # 方法1: 按照标准格式解析（保留向后兼容性）
        try:
            lines = processed_response.split('\n') if processed_response is not None and isinstance(processed_response, str) else []

            for line in lines:
                # 确保line不是None且是字符串类型
                if line is None or not isinstance(line, str):
                    continue
                if line.startswith('1. 相关性评分：') or '相关性评分：' in line:
                    # 提取评分
                    score_match = re.search(r'(\d+)/10分', line)
                    if score_match:
                        try:
                            result['relevance_score'] = int(score_match.group(1))
                        except (ValueError, IndexError):
                            pass  # 如果无法转换为整数，保持默认值
                elif line.startswith('2. 是否适合引用：') or '是否适合引用：' in line:
                    # 提取是否适合引用
                    if '是' in line and '不' not in line.replace('是', ''):
                        result['is_suitable_for_citation'] = True
                    elif '否' in line:
                        result['is_suitable_for_citation'] = False
                elif line.startswith('3. 简要依据：') or '简要依据：' in line:
                    # 提取简要依据
                    if line and '：' in line and isinstance(line, str) and len(line.split('：', 1)) > 1:
                        result['brief_basis'] = line.split('：', 1)[1].strip()
                    else:
                        result['brief_basis'] = line.replace('3. 简要依据：', '').replace('简要依据：', '').strip() if line and isinstance(line, str) else '简要依据未找到'
                elif detailed and '评分理由' in line:
                    # 提取详细理由（仅在详细分析中）
                    if line and '：' in line and isinstance(line, str) and len(line.split('：', 1)) > 1:
                        result['detailed_reasoning'] = line.split('：', 1)[1].strip()
                    else:
                        result['detailed_reasoning'] = line.replace('评分理由：', '').replace('理由：', '').strip() if line and isinstance(line, str) else '详细理由未找到'
        except Exception as e:
            print(f"按标准格式解析AI响应时发生异常: {e}")

        # 方法2: 如果标准格式解析失败，尝试更灵活的解析
        try:
            if result['relevance_score'] == 0 and (not result['brief_basis'] or result['brief_basis'] in ['简要依据未找到', '详细理由未找到']):
                # 尝试从整个响应中提取评分
                # 查找 "X/10分" 或 "X分" 的模式
                score_pattern = r'(\d{1,2}(?:\.\d{1,2})?)/10分|评分[:：]?\s*(\d{1,2}(?:\.\d{1,2})?)|相关性[:：]?\s*(\d{1,2}(?:\.\d{1,2})?)分'
                score_matches = re.findall(score_pattern, processed_response)
                for match in score_matches:
                    # match 是一个元组，包含所有捕获组
                    for group in match:
                        if group and group.strip():
                            try:
                                score = float(group)
                                result['relevance_score'] = min(int(score), 10)  # 评分最高为10
                                break
                            except ValueError:
                                continue
                    if result['relevance_score'] > 0:
                        break

                # 如果上面没找到评分，尝试更宽松的模式
                if result['relevance_score'] == 0:
                    loose_score_match = re.search(r'(\d{1,2})[分/]', processed_response)
                    if loose_score_match:
                        try:
                            score_val = int(loose_score_match.group(1))
                            result['relevance_score'] = min(score_val, 10)
                        except (ValueError, IndexError):
                            pass  # 如果无法转换为整数，保持默认值

                # 查找是否适合引用 - 使用更精确的模式
                yes_no_pattern = r'(是否适合引用|适合引用)[：:]?\s*(是|否|适合|不适合|推荐|不推荐)'
                yes_no_match = re.search(yes_no_pattern, processed_response)
                if yes_no_match:
                    decision_part = yes_no_match.group(2)
                    positive_keywords = ['是', '适合', '推荐', '可以', '建议']
                    negative_keywords = ['否', '不适合', '不推荐', '不建议', '不相关']

                    if any(kw in decision_part for kw in positive_keywords):
                        result['is_suitable_for_citation'] = True
                    elif any(kw in decision_part for kw in negative_keywords):
                        result['is_suitable_for_citation'] = False
                else:
                    # 如果没有找到明确的"是否适合引用"部分，单独检查关键词
                    if any(kw in processed_response for kw in ['适合', '推荐', '可以', '建议']):
                        result['is_suitable_for_citation'] = True
                    if any(kw in processed_response for kw in ['不适合', '不推荐', '不建议', '不相关', '否']):
                        result['is_suitable_for_citation'] = False

                # 查找简要依据 - 使用更广泛的模式
                basis_patterns = [
                    r'(简要依据|依据|理由|原因|分析|说明)[：:]?\s*([^\n\r。！？.!?]*[。！？.!?]|.*)',
                    r'(因为|由于|基于)[^，。！？.!?，。！？.!?]*[，。！？.!?]',
                    r'[^。！？.!?]*?(相关|引用|适合|不.*相关|不.*适合)[^。！？.!?]*[。！？.!?]'
                ]

                for pattern in basis_patterns:
                    match = re.search(pattern, processed_response)
                    if match:
                        if len(match.groups()) > 1:
                            result['brief_basis'] = match.group(2).strip()
                        else:
                            result['brief_basis'] = match.group(0).strip()
                        break

                # 如果仍然没有找到简要依据，尝试提取包含关键信息的句子
                if not result['brief_basis'] or result['brief_basis'] in ['简要依据未找到', '详细理由未找到']:
                    sentences = re.split(r'[。！？.!?]', processed_response)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10 and any(kw in sentence for kw in ['相关', '引用', '适合', '不', '因为', '所以', '由于']):
                            result['brief_basis'] = sentence
                            break

            # 如果仍然没有找到简要依据，使用AI响应的前100个字符
            if not result['brief_basis'] or result['brief_basis'] in ['简要依据未找到', '详细理由未找到']:
                result['brief_basis'] = processed_response[:100].strip() + "..." if len(processed_response) > 100 else processed_response.strip()

            # 确保布尔值被正确设置
            if result['relevance_score'] > 5:
                result['is_suitable_for_citation'] = True
            elif result['relevance_score'] <= 3:
                result['is_suitable_for_citation'] = False
        except Exception as e:
            print(f"灵活解析AI响应时发生异常: {e}")
            # 如果所有解析都失败，使用原始响应作为简要依据
            result['brief_basis'] = processed_response[:100].strip() + "..." if len(processed_response) > 100 else processed_response.strip()

        return result