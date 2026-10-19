# PaperCheck 运行说明

PaperCheck 在本地提取论文引用、参考文献和上下文，再由当前模型检查引用是否支持论述。`assets/paperchecker-rules/` 只保留文档提取、引用匹配、UCAS / GB/T 7714 规则和报告结构。

## 安装依赖

从本技能目录运行：

```bash
python -m pip install -r assets/paperchecker-rules/requirements.txt
python scripts/check_papercheck_env.py
```

处理 PDF 时另装本地解析依赖：

```bash
python -m pip install -r assets/paperchecker-rules/requirements-pdf.txt
```

普通检查不需要外部模型密钥。技能包不包含网页、API 服务、账号、短信或部署工具。上游规则的 MIT 许可保存在 `assets/paperchecker-rules/LICENSE`。

## 执行检查

```bash
python scripts/progressive_papercheck.py paper.docx --mode subjective --out evidence.json --report report.md
```

将文件名换成实际路径，也可输入 PDF。三个模式共用本地规则：

- `quick`：结构提取、引用匹配与格式规则。
- `subjective`：默认；增加供当前模型审阅的引用、上下文与参考文献对应项。
- `full`：在此基础上整理原文和来源核验范围。

证据 JSON 包含引用编号、范围展开、参考文献、缺失引用、未使用条目和 `needs_model_review` 标记。Markdown 报告包含实际解析方式、规则结果、模型审阅队列与限制。格式规则不自动适用于 AAS、MNRAS 或全部天文期刊。

## PDF 解析

配置 `MINERU_API_KEY` 后，PDF 解析可使用 MinerU。也可在本机 `assets/paperchecker-rules/config/config.json` 的 `mineru_config.api_key` 中配置；不要将私人配置提交到仓库。

未配置或 MinerU 不可用时，依次尝试 `pymupdf4llm.to_markdown()` 和 PyMuPDF 的逐页文本提取。环境检查会报告依赖和配置状态，输出会标明回退方式。文本层 PDF 可以继续处理；扫描件、多栏、表格、公式和跨页参考文献仍需人工核对或 OCR / 版面解析。

没有被引论文的原文或已核验来源时，只能判断题名、引用上下文与参考文献的对应关系，不能声称核对了全文。

## 自检与打包

```bash
python scripts/quick_validate.py
```

包内保留入口说明、脚本、参考文件、测试和精简后的本地规则。依赖目录、缓存、私人配置、上传论文和生成报告均不打入技能包。完整的早期规则服务可从 Git 历史找回。
