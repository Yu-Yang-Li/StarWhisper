from .base_extractor import BaseExtractor
from .pdf_extractor import PDFExtractor
from models.document import Document


class MarkdownExtractor(BaseExtractor):
    """Markdown 文档提取器"""

    def validate_file(self, file_path: str) -> bool:
        return file_path.lower().endswith('.md')

    def extract(self, file_path: str) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        paragraphs = md_content.split('\n') if md_content else []
        tables_content = []
        for line in paragraphs:
            if '|' in line and line.strip().startswith('|'):
                tables_content.append(line.strip())

        # 复用 PDF 路径已验证的正文/参考文献分段与块提取逻辑
        helper = PDFExtractor()
        citations = helper._extract_citations(paragraphs, md_content, file_path)
        references = helper._extract_references(paragraphs)

        return Document(
            content=paragraphs,
            tables=tables_content,
            citations=citations,
            references=references,
            metadata={'file_type': 'md', 'file_path': file_path}
        )
