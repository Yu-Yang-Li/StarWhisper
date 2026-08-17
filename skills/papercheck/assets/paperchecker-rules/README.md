# TaShan-PaperChecker

**他山·论文引用合规检查** — 基于规则引擎的学术论文引用格式检查工具，支持 GB/T 7714 标准及 UCAS（中国科学院大学）学位论文规范。

> 另见 AI 版本：[PaperCheck](https://github.com/TashanGKD/PaperCheck) — 通过大模型对论文内容与参考文献的相关性、完整性进行综合分析。

**在线体验**：[cite.tashan.chat](https://cite.tashan.chat)

---

## 特性

- **格式规则检查**：根据 GB/T 7714-2015 和 UCAS 规范，逐条核查参考文献格式
- **引用匹配**：检测正文引用与参考文献列表的对应关系（支持著者-出版年制、顺序编码制）
- **问题分级**：强规则 / 启发规则 × 高 / 中 / 低置信度，输出可机读的结构化 JSON
- **两套前端**：
  - `front/tashan-ui/` — 他山设计系统（国风主题），纯 HTML+JS，无需构建
  - `front/web/` — 原始控制台前端
- **REST API v2**：统一契约格式，方便脚本调用或对接其他工具

---

## 快速开始

### 依赖

- Python 3.10+
- 推荐使用虚拟环境

### 安装

```bash
git clone https://github.com/TashanGKD/TaShan-PaperChecker.git
cd TaShan-PaperChecker
pip install -r requirements.txt
```

### 启动

```bash
python run_server.py
# 或者指定端口
SERVER_PORT=3950 python run_server.py
```

服务启动后访问：

- `http://localhost:8000/ui/` — 他山设计系统前端（国风主题）
- `http://localhost:8000/frontend/` — 原始控制台前端
- `http://localhost:8000/api/health` — 健康检查
- `http://localhost:8000/docs` — Swagger API 文档

---

## API

### 健康检查

```
GET /api/health
```

### 上传检查（推荐）

```
POST /api/v2/analysis/report
Content-Type: multipart/form-data

file              上传的论文文件 (.docx / .doc / .pdf)
author_format     "full"（著者-出版年制）| "abbrev"（顺序编码制）  默认 full
citation_standard "legacy"（通用格式）| "ucas"（UCAS 规范）        默认 legacy
```

**响应示例**

```json
{
  "contract_version": "2.0.0",
  "run": {
    "run_id": "analysis.report_1745000000",
    "operation": "analysis.report",
    "status": "succeeded",
    "duration_ms": 1832
  },
  "summary": {
    "total_citations": 42,
    "total_references": 45,
    "match_rate": "93.3%",
    "high_confidence_issue_count": 3,
    "reference_format_issue_count": 5,
    "citation_style_issue_count": 2
  },
  "issues": {
    "reference_format_issues": [...],
    "citation_style_issues": [...],
    "unused_references": [...],
    "unmatched_citations": [...]
  },
  "evidence": { ... },
  "error": null
}
```

### 其他端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v2/workspace/upload` | 仅上传，不分析 |
| GET  | `/api/v2/workspace/files` | 列出已上传文件 |
| POST | `/api/v2/analysis/report-from-path` | 对已上传文件分析 |
| GET  | `/api/health` | 服务健康检查 |

完整接口文档：启动后访问 `http://localhost:8000/docs`

---

## 部署

### 单机部署

```bash
# 后台启动（推荐）
SERVER_HOST=0.0.0.0 SERVER_PORT=3950 SERVER_RELOAD=false \
  nohup python run_server.py > /var/log/tashan-paperchecker.log 2>&1 &
```

### Nginx 反代

```nginx
server {
    listen 80;
    server_name cite.example.com;
    client_max_body_size 60M;

    location /api/ {
        proxy_pass http://127.0.0.1:3950;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }

    location / {
        proxy_pass http://127.0.0.1:3950;
        proxy_set_header Host $host;
    }
}
```

### Docker

```bash
docker build -t tashan-paperchecker .
docker run -p 8000:8000 tashan-paperchecker
```

---

## 项目结构

```
TaShan-PaperChecker/
├── app/
│   └── main.py               FastAPI 应用入口
├── core/
│   ├── checker/
│   │   └── citation_checker.py   规则引擎（GB/T 7714 + UCAS）
│   ├── extractor/            文档解析（docx / pdf）
│   └── processors/           引用处理管线
├── contracts/
│   ├── v2_contract.py        API 响应契约构建
│   └── report_contract.py    问题分类与汇总
├── services/
│   ├── analysis_service.py   分析服务
│   └── workspace_service.py  文件工作区管理
├── front/
│   ├── tashan-ui/            他山设计系统前端（国风主题）
│   │   ├── index.html
│   │   └── assets/colors_and_type.css
│   └── web/                  原始控制台前端
├── config/config.py
├── run_server.py
└── requirements.txt
```

---

## 开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动（热重载）
SERVER_RELOAD=true python run_server.py

# 运行测试
pytest tests/
```

---

## 相关项目

| 项目 | 说明 | 地址 |
|------|------|------|
| PaperCheck | AI 版本，大模型综合分析 | [github.com/TashanGKD/PaperCheck](https://github.com/TashanGKD/PaperCheck) |
| TaShan-PaperChecker | 本项目，规则引擎版本 | [github.com/TashanGKD/TaShan-PaperChecker](https://github.com/TashanGKD/TaShan-PaperChecker) |

两个项目定位不同，互补而非替代：
- **规则版（本项目）**：速度快、可解释、适合批量处理，聚焦引用格式
- **AI 版**：理解语义、适合深度审阅，覆盖内容相关性与完整性

---

## License

MIT
