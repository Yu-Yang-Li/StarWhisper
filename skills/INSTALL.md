# 安装与运行环境

[技能目录](README.md) · [仓库结构](../docs/architecture.md) · [来源与许可](NOTICE.md)

安装脚本会复制技能目录，不会安装科研项目、Python 依赖、模型权重或外部服务密钥。先查看技能清单和预计安装位置，再选择需要的分组。

## 安装到 Codex / Cursor

以下命令从仓库根目录运行。Windows 使用 PowerShell：

```powershell
powershell -File .\skills\install.ps1 -List
powershell -File .\skills\install.ps1 -DryRun
powershell -File .\skills\install.ps1
```

Linux / macOS 使用 Bash：

```bash
bash ./skills/install.sh --list
bash ./skills/install.sh --dry-run
bash ./skills/install.sh
```

默认将全部 17 个技能安装到 `~/.codex/skills` 与 `~/.cursor/skills`。安装会替换同名目录；目录有本地修改时先备份。

| 选项 | PowerShell | Bash |
| --- | --- | --- |
| 只安装 4 个原生技能 | `-Set native` | `--set native` |
| 只安装 13 个改编技能 | `-Set research` | `--set research` |
| 安装全部技能 | `-Set all` | `--set all` |
| 只安装到 Codex | `-Target codex` | `--target codex` |
| 只安装到 Cursor | `-Target cursor` | `--target cursor` |
| 两处都安装 | `-Target both` | `--target both` |

例如，只安装原生技能到 Codex：

```powershell
powershell -File .\skills\install.ps1 -Set native -Target codex
```

`SKILL.md`、参考文件、脚本和模板会一起复制。PPT 脚本使用同级的科研绘图技能，安装 `research` 或 `all` 时两者都包含在内。

## 让技能找到仓库数据

从仓库内运行时，原生脚本会向上查找 `projects/` 下的项目文件，也兼容旧版顶层目录。安装到其他位置后，可以设置 `STARWHISPER_ROOT`，指向包含 `projects/` 和 `skills/` 的仓库根目录。从仓库根目录设置当前 PowerShell 会话，并保存给之后新开的终端：

```powershell
$env:STARWHISPER_ROOT = (Get-Location).Path
setx STARWHISPER_ROOT "$env:STARWHISPER_ROOT"
```

Linux / macOS 使用 `export STARWHISPER_ROOT="$PWD"`；长期保留时写入自己的 shell 配置。也可以按技能说明使用 `--csv` 或 `--config` 指定实际文件。

不同工具的默认文件与回退行为见[仓库结构](../docs/architecture.md#离线工具怎样读取项目材料)。SN Clock 缺少预测表时会退出；夜计划使用示例配置时会标明来源。先确认输出中的输入来源，再解释结果。

## Python 依赖

4 个原生技能需要 Python 3.10 或更新版本，只用标准库。运行回归测试才需要 `pytest`。其他技能按所需步骤安装依赖，不必一次安装所有项目的环境。

| 任务 | 依赖说明 |
| --- | --- |
| 实验设计 | [experiment-design/requirements.txt](experiment-design/requirements.txt) |
| 统计分析 | [statistical-analysis/requirements.txt](statistical-analysis/requirements.txt) |
| 论文全文审查 | [thesis-audit-reviewer/requirements.txt](thesis-audit-reviewer/requirements.txt) |
| PPT 打包与检查 | [visual-deck-builder/scripts/requirements.txt](visual-deck-builder/scripts/requirements.txt) |
| 引用规则检查 | [papercheck 的 requirements.txt](papercheck/assets/paperchecker-rules/requirements.txt) |
| PaperCheck 本地 PDF 解析 | [requirements-pdf.txt](papercheck/assets/paperchecker-rules/requirements-pdf.txt)；无需启动网页或 API 服务 |
| 基线模型模板 | [模型模板说明](research-baseline-builder/references/model-templates.md)，各脚本开头也列有依赖 |

科研训练代码的环境以各项目 README 为准。技能安装不会自动安装 NINA、FTP、MQTT 或图像推断模型。

## 外部服务

| 变量 | 用在哪一步 | 没有时怎样处理 |
| --- | --- | --- |
| `ADS_API_TOKEN` | NASA ADS 检索 | dry-run 只查看请求；实际检索可回退到 arXiv `astro-ph` |
| `GIIISP_AUTH_TOKEN` | Giiisp 补充检索与图片服务 | 准备请求，说明没有执行检索或生成 |
| `MINERU_API_TOKEN` | 在线论文 PDF 解析 | 按任务选择本地解析，并说明缺失内容 |
| `DASHSCOPE_API_KEY` | 部分内容生成与视觉审查 | 使用可用方法并记录；没有执行的审查不能标为通过 |
| `GIIISP_IMAGE_SKILL_DIR` | PPT 脚本定位配套绘图技能 | 默认使用相邻目录；单独存放时指定实际目录 |

PPT 修订流程支持 `--style-contract`；传入样式文件时，五项科研图检查和三项样式检查均需 PASS。缺项、不确定或服务不可用时保留问题记录。本地文件和格式检查通过不代表生成与视觉审查已经完成。

密钥从环境变量读取，不放进 Git、示例文件或报告。历史实测记录、示例和 dry-run 不表示外部服务现在可用；按本次执行结果报告。
