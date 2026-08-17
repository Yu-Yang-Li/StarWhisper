# PaperChecker 部署说明

## 项目概述
这是一个文档检查工具，包含后端API服务和前端Web界面。

## 部署方式

### 本地部署
1. 克隆项目代码
2. 安装依赖：`pip install -r requirements.txt`
3. 启动服务：`python run_server.py`
4. 服务将在默认端口8002上启动

### 生产环境部署
对于生产环境，建议：
- 使用反向代理（如Nginx）
- 配置进程管理器（如Supervisor或systemd）
- 设置适当的日志管理和监控

## 技术架构
- 前端：HTML/JavaScript（位于 front/web 目录）
- 后端：Python/FastAPI（运行在配置的端口上）
- 可选反向代理：Nginx（用于生产环境）

## 配置选项
服务配置可通过 `config/config.py` 文件或环境变量进行调整：
- `SERVER_HOST`: 服务器主机地址（默认：0.0.0.0）
- `SERVER_PORT`: 服务器端口（默认：8002）
- `MAX_UPLOAD_SIZE`: 最大上传文件大小（默认：10MB）