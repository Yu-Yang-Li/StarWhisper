---
name: thesis-audit-reviewer
description: "审查天文论文或学位论文的方法、数据、公式、图表、引用与结论；生成可定位、有原文和依据的批注或审核报告，并记录未能核验的部分。"
---

# 论文与学位论文审查

检查方法、数据、公式、图表、引用和结论之间是否一致。每条正式意见都要能定位到原文，并说明依据与具体修改要求。

## 先确定范围

确认待审文件、用户给出的检查清单和输出要求。一般修改意见与正式全文审核的深度不同；用户只讨论流程时，不运行完整审查。

天文对象、坐标、测光和时间系统的检查见 [astronomy.md](astronomy.md)。完整审查前阅读 [操作流程](references/audit-workflow.md)、[默认审查依据](references/default_review_sources.md) 和 [输入输出规则](references/input_output_policy.md)。用户提供的清单优先。

## 审查顺序

1. 记录原始文件、页数和解析方式，保留原文件。
2. 建立页面、章节、图表、公式、引用、方法和关键数字的清单。
3. 先核对摘要、方法前提、核心计算与结论，再检查其余内容。
4. 高风险问题回看 PDF 页面或 DOCX 原始对象；解析文本不能替代原图。
5. 对已检查、发现问题和无法判断的部分分别记录，不把缺页或未知写成通过。
6. 汇总意见，检查覆盖范围，并运行报告校验。

## 文件与工具

以下命令在本技能目录中运行：

```bash
python scripts/doctor.py
python scripts/init_audit_workspace.py --out subprojects/thesis_audit --paper-id paper01 --title "论文题名" --source-file paper.pdf
```

解析、事实台账和报告命令见 [操作流程](references/audit-workflow.md)。PDF 通常输出 PDF 报告，DOCX 通常输出带批注的 DOCX；格式与用户要求冲突时先以用户要求为准。需要实际文档批注时使用环境中可用的文档工具。

保密、未发表或不允许上传的论文使用本地解析；未经允许不交给在线解析服务。MinerU 的密钥只从环境、交互输入或 stdin 读取，不写进命令与报告。DOCX 输入需检查公式和图片是否完整，见 [完整性检查](references/docx_integrity_gate.md)。

## 一条意见应怎样写

每条包含位置、原文摘录、审核意见、证据依据、具体修改要求、优先级和确定性。依据不足的线索留在工作记录，不进入正式问题清单。

用 [标准报告模板](references/standard_report_template.md) 和 [批注措辞](references/comment_language.md) 写报告，按 [完成条件](references/completion_gate.md) 检查覆盖情况。校验命令为：

```bash
python scripts/validate_audit_report.py --report outputs/paper01/审查报告.md --strict
```

报告路径换成实际文件。交付时说明哪些内容已核验、哪些受限，附上报告与工作文件路径；多篇论文先分别满足单篇检查要求，再汇总。
