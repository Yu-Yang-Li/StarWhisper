---
name: visual-deck-builder
description: "根据天文论文、巡天材料或研究笔记制作和修改 PPT，保留科学信息与来源，生成页面预览并检查内容、文字和样式。"
---

# 天文汇报与 PPT

从论文、巡天材料、笔记或已有页面制作汇报，保留科学信息与来源。默认将完整页面图片打包为 PPTX；需要可编辑元素时选择重建流程，并说明实际可编辑的部分。

## 先确定这次要做什么

| 输入 | 做法 | 需要时阅读 |
| --- | --- | --- |
| 主题或提纲 | 整理受众、页数和内容，标出尚无来源的例子 | [生成流程](references/deck-workflow.md) |
| 论文、数据或笔记 | 从原材料取内容，保留数字、术语和引用 | [输入类型](references/input-routing.md) |
| 模板或参考页面 | 参考指定样式，避免带入无关内容 | [样式约定](references/aesthetic-generation.md) |
| 已有 PPT 或图片 | 只改指定页面，保留其余内容 | [检查与修改](references/review-and-repair.md) |
| 必须可编辑 | 先判断哪些文字、图形或图表能重建 | [可编辑重建](references/editable-reconstruction-route.md) |

天文资料与结果性质见 [astronomy.md](astronomy.md)。其他 PPT 技能只在当前环境确实存在、且任务需要时使用，不作为这个包的必装依赖。

## 一次完整流程

1. 确认材料、用途、受众、页数、语言和修改范围。
2. 写 `slide_spec.json`，逐页记录主题、可见文字、来源、页面用途与布局。
3. 写 `visual_style_contract.json`，明确配色、字体、间距和图标规则。
4. 生成页面与预览；每页文字、数字和科学结论仍以原材料为准。
5. 核对内容、可读性、伪影和样式，再打包 PPTX 并检查页数、图片与预览。

命令与字段细则见 [生成流程](references/deck-workflow.md)。无模板时可先列出已有样式：

```bash
python scripts/build_style_contract.py --list
```

以上命令在本技能目录中运行；打包与检查依赖 [Pillow、python-pptx](scripts/requirements.txt)。完整图像生成与语义审查依赖相邻的 [科研绘图技能](../giiisp-scientific-image-generation/SKILL.md) 及对应服务密钥；单独存放该技能时用 `GIIISP_IMAGE_SKILL_DIR` 指定目录。

## 修改与检查

先留下逐页问题与修订建议。用户已要求修改哪些页，就按该范围执行；未要求重生成时，不自动重做页面。修订脚本的 `--slides 01,03` 只处理选中的失败页，详见 [检查与修改](references/review-and-repair.md)。

候选页面通过审查后才替换原图。传入 `--style-contract` 时，五项科研图检查与三项样式检查均需 PASS；缺项、不确定或服务不可用时保留问题，不能声称复查通过。页面有四页及以上时需有正文型页面，避免连续重复同一种版式；详细检查项见 [生成流程](references/deck-workflow.md) 的 QA 部分。

## 会得到什么

一份 PPTX、页面图片与预览、内容和样式说明、逐页审查记录，以及实际使用的素材来源。每轮放在独立 `visual_deck_runs/<时间>_<主题>/` 目录中，修改时保留原图与未通过的候选图。

进度只报告实际发生的步骤和具体问题，不向用户输出原始服务日志。进度字段、时限、相关项目的对照记录见 [运行说明](references/runtime-notes.md)。
