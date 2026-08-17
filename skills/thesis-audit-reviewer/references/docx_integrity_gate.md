# DOCX 完整性门禁

DOCX 不能只用 `python-docx paragraph.text` 或命令行预览来读取。Word 论文常包含 OOXML Math 公式、文本框、图片、批注、修订、脚注、域代码和表格嵌套，这些对象可能在普通文本抽取中丢失。

## 一、门禁目标

在进入正式审阅前，必须确认智能体实际可见：

- 正文段落。
- 表格。
- OOXML Math 公式。
- 图片/图题/图中文字。
- 脚注、尾注、批注、修订。
- 目录、交叉引用、域代码。

若不可见，必须转换、渲染或标记 blocked，不得直接给出 `pass`。

## 二、必做步骤

1. 运行 `scripts/docx_integrity_scan.py --docx <paper.docx>`，记录：
   - 段落数、表格数、图片数。
   - OOXML Math 对象数及所在段落数。
   - 脚注、尾注、批注、修订、域代码、超链接数量。
2. 若存在公式、图片、文本框或复杂表格：
   - 用 Word/LibreOffice/textutil/Documents 能力生成可视化版本，或直接打开原 DOCX。
   - 对高风险公式、图、表回看渲染结果。
3. 对长文正文按章或固定段落数切分，不依赖 Bash/Read 的超长输出预览。
4. 在覆盖报告中记录实际阅读范围和未能读取的对象。

## 三、禁止事项

- 禁止把工具输出预览当成全文阅读。
- 禁止只读目录和表格后声称逐行审阅正文。
- 禁止在未查看公式渲染的情况下判断公式完整。
- 禁止把 DOCX 抽取失败造成的公式空白直接判为作者错误；应先回看原文或渲染版。
- 禁止把“对象已登记”等同于“对象已检查”。

## 四、DOCX 状态口径

- `visible`：对象已在文本或渲染版中可见。
- `needs_render_check`：文本抽取不足，需要回看 DOCX/PDF 渲染。
- `needs_ooxml_check`：需要检查 OOXML 中公式、域代码、批注或修订。
- `blocked`：当前工具无法读取或渲染，需用户提供 PDF、截图或可编辑源文件。

## 五、与批注版 DOCX 的关系

DOCX 输入默认应交付批注版 DOCX。若无法写入真实 Word 批注，必须交付可追溯批注清单并说明限制。批注清单不是替代审阅本体，仍需保留矩阵、问题库、事实账本和覆盖报告。

## 六、PDF 本地降级口径

PDF 输入在没有 MinerU 或无法上传时，先运行 `scripts/pdf_local_fallback_extract.py`。状态文件 `pdf_fallback_status.json` 中的 `method`、`quality`、`warnings` 必须写入源文件登记或覆盖报告。

- `local_markdown_fallback`：`pymupdf4llm` 成功，适合搭建章节、段落和初步对象账本。
- `local_text_layer`：`fitz` 文本层可用，但版面结构弱于 MinerU。
- `partial_text_layer`：部分页面有文本，必须结合渲染页复核。
- `likely_scanned_or_image_pdf`：疑似扫描件或图片 PDF，不能直接做完整审阅；应 OCR、回看原 PDF、改用更强解析方式，或标记 blocked。

本地 fallback 不等于版面审查已完成。公式、图表、复杂表格、跨页参考文献和多栏排版仍要回看原 PDF、改用更强解析方式，或标记 blocked/residual risk。
