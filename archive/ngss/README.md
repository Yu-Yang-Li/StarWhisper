# NGSS 历史运行材料

[归档目录](../README.md) · [当前 NGSS 项目](../../projects/ngss/README.md)

- `runningLog/`：仓库原有的四份 n8n 事件日志。
- `compiled-cache/`：原 `src/util/Decorators/Logging/__pycache__/` 中的文件，包含 Python 3.10–3.13 的编译缓存。

对应的 Python 源文件仍在 NGSS 项目的 `src/util/Decorators/Logging/`。启动时由当前 Python 环境按需生成缓存。历史文件保留原始字节，不作为当前观测状态或运行输入。
