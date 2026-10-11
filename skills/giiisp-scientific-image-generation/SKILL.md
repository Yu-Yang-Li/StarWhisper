---
name: giiisp-scientific-image-generation
description: "根据论文、流程说明或参考图生成和修改科研示意图，保存版本与审查结果。光变曲线、光谱等具有实际数据刻度的图应由数据绘制。"
---

# 科研示意图

根据论文段落、流程说明或参考图生成和修改示意图，保留每版图片、请求和审查记录。光变曲线、光谱等有实际数据刻度的图应使用原始数据绘制，不能让生成模型补出测量结果。

## 先写清图要表达什么

确定用途、科学关系、必须出现的标签、画幅和风格。天文图的标注与结果性质见 [astronomy.md](astronomy.md)。参考图用于保留结构、借用元素、润色草图还是编辑当前图，要明确记录。

图的事实来源是 `figure_spec.json`：图题、意图、标签、布局、风格、参考图用途，以及允许和禁止的改动。详细字段与调用方法见 [生成与检查](references/generation-workflow.md)。

## 查看请求，或开始生成

以下命令在本技能目录中运行。先检查请求，不调用图片服务：

```bash
python scripts/dry_run_scientific_image.py --prompt "画一张示意图：测光数据进入年龄估计模型，输出带不确定性的候选排序"
```

使用完整脚本流程时，需要实际可用的图片服务、`GIIISP_AUTH_TOKEN` 和视觉审查所需的 `DASHSCOPE_API_KEY`。参数与示例见 [生成与检查](references/generation-workflow.md)；环境里有其他可用图像工具时，记录实际使用的工具与审查方式。

生成后检查文件、尺寸与格式，再核对科学内容、标签、文字、伪影和布局。机器检查通过只说明图片文件可读。缺少密钥、接口失败或审查未完成时保留 blocked 记录，不补写通过结论。

## 修改已有图片

先看上一版、原始简报和用户反馈，只修改指定内容。每次创建新 run，保留父版本、源图哈希、反馈和允许改动项，不能覆盖上一轮。

脚本完整流程最多自动修订一轮，保留 `critic_suggestions` 与 `revised_description`；第二版复查后停止，由用户决定是否继续。用户进一步要求的修改另建版本。具体命令见 [改图、图包与候选选择](references/edit-and-packaging.md)。

## PPT 页面审查

`semantic_review_dashscope.py` 可接受 `--style-contract visual_style_contract.json`，在五项科研图检查外增加三项页面样式检查。八项均为 PASS 才可判为通过；缺项、UNCERTAIN 或无效样式文件均不能接受候选图。普通科研图无需这个参数。

## 会得到什么

run 目录中保留 `figure_spec.json`、脱敏请求与响应、图片、`check.json`、`semantic_review.json`、`manual_review.md` 和 `figure_manifest.json`。代理检查摘要不等于人工签核。返回图片路径、检查方式、实际结果和仍需修改的部分；多图汇总与候选选择见 [改图与打包](references/edit-and-packaging.md)。

提示词设计来源、进度示例和早期试运行记录见 [设计说明](references/design-notes.md)，需要时再读；它们不能证明当前接口可用。
