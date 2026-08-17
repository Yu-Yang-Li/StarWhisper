import os
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
        # 构造AI提示词
        document_title = document.metadata.get('title', '未知标题') if hasattr(document, 'metadata') else '未知标题'
        prompt = f"""请分析以下文献引用的相关性，仅基于提供的标题信息进行判断：
【任务类型】{task_type}相关性判断
【引用文献标题】{document_title}
【对比内容】{target_content}
请按以下格式回答：
1. 相关性评分：X/10分
2. 是否适合引用：是/否
3. 简要依据：不超过50字
注意：由于仅基于标题判断，不要臆测文献内容。"""

        # 调用AI生成结果
        print(f"DEBUG: Calling ai_generate with prompt: {prompt[:100]}...")  # 调试信息
        ai_response = ai_generate(prompt)
        print(f"DEBUG: ai_generate returned: {type(ai_response)} - {ai_response}")  # 调试信息

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

        # 构造AI提示词
        prompt = f"""请全面分析以下文献引用的相关性，基于提供的完整文献内容和对比内容：
【任务类型】{task_type}相关性判断
【引用文献】{full_content}
【对比内容】{target_content}
请按以下格式回答：
1. 相关性评分：X/10分，并简述评分理由
2. 是否适合引用：是/否
3. 简要依据：不超过50字
注意：请严格基于文献内容分析，避免超出文本的推断。"""

        # 调用AI生成结果
        print(f"DEBUG: Calling ai_generate with prompt for accurate check...")  # 调试信息
        ai_response = ai_generate(prompt)
        print(f"DEBUG: ai_generate returned for accurate check: {type(ai_response)} - {ai_response}")  # 调试信息

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
        print(f"DEBUG: _get_full_content called with document: {type(document)}")  # 调试信息
        # 如果文档已经有完整内容，则直接返回
        if hasattr(document, 'full_text') and document.full_text:
            return document.full_text

        # 检查文档的metadata中是否有文件路径
        file_path = None
        if hasattr(document, 'metadata') and 'file_path' in document.metadata:
            file_path = document.metadata['file_path']
        elif hasattr(document, 'file_path'):
            file_path = document.file_path

        print(f"DEBUG: File path found: {file_path}")  # 调试信息

        # 如果有文件路径，使用适当的提取器提取内容
        if file_path and os.path.exists(file_path):
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')

            if ext in ['doc', 'docx']:
                from core.extractor.word_extractor import WordExtractor
                extractor = WordExtractor()
            elif ext == 'pdf':
                # 使用改进的PDF提取器，进行本地处理，避免API调用
                from core.extractor.pdf_extractor import PDFExtractor
                extractor = PDFExtractor()
            else:
                # 如果文件类型不支持，尝试获取文档标题用于AI搜索
                document_title = document.metadata.get('title', '') if hasattr(document, 'metadata') else ''
                return document_title if document_title is not None else "无法获取文献内容"

            try:
                # 提取文档内容
                print(f"DEBUG: Extracting content from: {file_path}")  # 调试信息
                extracted_document = extractor.extract(file_path)
                print(f"DEBUG: Extracted document: {type(extracted_document)}, content type: {type(extracted_document.content)}")  # 调试信息
                
                # 将提取的段落合并为完整内容
                if extracted_document.content and extracted_document.content is not None:
                    # 过滤掉 None 值并确保所有元素都是字符串
                    filtered_content = [str(item) if item is not None else "" for item in extracted_document.content]
                    full_content = '\n'.join(filtered_content)
                else:
                    full_content = "文档内容为空"
                print(f"DEBUG: Full content: {full_content[:100]}...")  # 调试信息
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
            # 构造AI搜索提示词
            prompt = f"""请根据以下文献标题，提供该文献的摘要或主要内容概述：

文献标题：{title}

请提供：
1. 文献的摘要或主要内容概述
2. 文献的主要研究方法
3. 文献的主要结论

如果无法找到该文献的具体信息，请返回"未找到相关文献信息"。
"""
            # 调用AI生成内容
            ai_response = ai_generate(prompt)

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
        return f"无法下载PDF: {pdf_url}"

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
        print(f"DEBUG: _parse_ai_response called with ai_response: {type(ai_response)}, value: {ai_response}")  # 调试信息
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
            print("DEBUG: ai_response is None")  # 调试信息
            return result

        # 确保ai_response是字符串类型
        if not isinstance(ai_response, str):
            result["brief_basis"] = f"AI响应类型错误，期望字符串，实际为{type(ai_response)}，无法进行相关性分析"
            print(f"DEBUG: ai_response is not a string: {type(ai_response)}")  # 调试信息
            return result

        # 解析AI响应中的各个部分
        lines = ai_response.split('\n') if ai_response is not None and isinstance(ai_response, str) else []
        print(f"DEBUG: Split ai_response into {len(lines)} lines")  # 调试信息
        for line in lines:
            print(f"DEBUG: Processing line: {line}")  # 调试信息
            # 确保line不是None且是字符串类型
            if line is None or not isinstance(line, str):
                print("DEBUG: Line is None or not string, skipping")  # 调试信息
                continue
            if line.startswith('1. 相关性评分：') or '相关性评分：' in line:
                # 提取评分
                import re
                score_match = re.search(r'(\d+)/10分', line)
                if score_match:
                    result['relevance_score'] = int(score_match.group(1))
            elif line.startswith('2. 是否适合引用：') or '是否适合引用：' in line:
                # 提取是否适合引用
                if '是' in line:
                    result['is_suitable_for_citation'] = True
            elif line.startswith('3. 简要依据：') or '简要依据：' in line:
                # 提取简要依据
                if line and '：' in line and isinstance(line, str) and len(line.split('：', 1)) > 1:
                    result['brief_basis'] = line.split('：', 1)[1]
                else:
                    result['brief_basis'] = line.replace('3. 简要依据：', '') if line and isinstance(line, str) else '简要依据未找到'
            elif detailed and '评分理由' in line:
                # 提取详细理由（仅在详细分析中）
                if line and '：' in line and isinstance(line, str) and len(line.split('：', 1)) > 1:
                    result['detailed_reasoning'] = line.split('：', 1)[1]
                else:
                    result['detailed_reasoning'] = line.replace('评分理由：', '') if line and isinstance(line, str) else '详细理由未找到'

        print(f"DEBUG: Parsed result: {result}")  # 调试信息
        return result