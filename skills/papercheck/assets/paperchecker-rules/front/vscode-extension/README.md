# PaperChecker VS Code 插件

该插件允许在 VS Code 内直接调用 PaperChecker 的 FastAPI 后端完成文档分析，并将后端返回的 JSON 报告立即转换为 Markdown 文件。

## 功能

- 一键启动/停止本地 PaperChecker 后端服务。
- 选择 Word 文档并调用 `/api/full-report` 生成分析结果。
- 自动将分析结果保存为 JSON 与 Markdown（Markdown 文件默认位于 `reports_md/`）。
- 独立命令支持将任意历史 JSON 报告转换为 Markdown。

## 使用步骤

1. 在 VS Code 中打开包含本项目的工作区。
2. 通过命令面板执行 `PaperChecker: 启动后端服务`（或在首次调用分析功能时按提示启动）。
3. 执行 `PaperChecker: 分析文档并生成报告`，选择待分析的 `.docx` 或 `.doc` 文件。
4. 命令执行完毕后，可在输出面板查看日志，并在 `reports/`、`reports_md/` 中查看生成文件。

## 配置项

通过 VS Code 设置搜索 `PaperChecker` 可调整以下选项：

- `paperchecker.serverUrl`：后端地址，默认为 `http://127.0.0.1:8000`。
- `paperchecker.pythonPath`：运行脚本所用的 Python 解释器。
- `paperchecker.jsonOutputDir`：JSON 报告输出目录（相对工作区）。
- `paperchecker.markdownOutputDir`：Markdown 报告输出目录（相对工作区）。

## 依赖

- 已安装的 Python（与项目依赖保持一致）。
- 项目根目录可直接运行 `python run_server.py`。

## 开发提示

插件的核心入口文件位于 `vscode-extension/extension.js`。如需扩展功能，可在该文件中增加新的 VS Code 命令并调用已有 Python 工具脚本。

