# NGSS · StarWhisper Telescope

近邻星系巡天（NGSS）的观测自动化代码，对应论文 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。程序负责制定观测计划、管理目标、连接 NINA，以及传输和处理观测数据。

[项目首页](../../README.md) · [研究文档](../../docs/research.md) · [离线配置检查](../../skills/starwhisper-night-plan/SKILL.md)

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/app/app2.py` | FastAPI 服务入口 |
| `src/module/` | 观测计划、目标处理、通信等模块 |
| `src/script/`、`src/util/` | 更新脚本与辅助函数 |
| `observe.yml` | Python 环境依赖 |
| `observe_config.json` | 观测配置 |
| `Make_Observation_Plan.json`、`NGSS_Agent.json` | n8n 工作流示例 |
| `observation Schedule/` | 观测计划样例 |

运行日志与 Python 编译缓存不随源码分发。服务日志按部署环境保存。

## 环境与启动

运行需要 NINA、FTP、x-opstep 和已配置的望远镜连接。先在 `projects/ngss/` 目录创建环境：

```bash
conda env create -f observe.yml
```

激活 `observe.yml` 中指定的环境，再从同一目录启动服务：

```bash
uvicorn src.app.app2:app --reload
```

这条命令默认使用端口 8000，接口说明可在 [本地 API 文档](http://127.0.0.1:8000/docs) 查看。`--reload` 用于开发调试；启动服务不代表设备连接和真实观测已经验证。

## 观测配置

程序读取 `observe_config.json` 的平铺字段。仓库中的示例为：

```json
{
  "inherit": true,
  "early_night": 0.5,
  "midnight": 2.0,
  "midmorning": 2.0,
  "early_morning": 2.0,
  "d_moon": 15,
  "FilterType": ["L"],
  "TotalExposureCount": 3,
  "ExposureTime": 120,
  "WaitTime": 1.0
}
```

| 字段 | 含义 |
| --- | --- |
| `inherit` | 是否继承前一日计划 |
| `early_night`、`midnight`、`midmorning`、`early_morning` | 各时段的目标选择时间窗，单位为小时 |
| `d_moon` | 最小月距，单位为度 |
| `FilterType` | 滤光片列表 |
| `TotalExposureCount` | 每个目标的曝光次数 |
| `ExposureTime` | 单次曝光时间，单位为秒 |
| `WaitTime` | 每个目标的额外指向与读出时间，单位为分钟 |

调整前可用 [`starwhisper-night-plan`](../../skills/starwhisper-night-plan/SKILL.md) 在本地检查字段、目标表和曝光预算。该技能读取文件，不连接设备。

## 设备与数据处理依赖

- 在 `src/module/UdpConnect.py` 中核对通信配置。
- 配置 [X-OPSTEP](https://github.com/Shinehope-Chen/X-OPSTEP/blob/master/README.md) 及其 astrometry.net、ASTAP、Scamp、SWarp、HOTPANTS 依赖。
- 安装 [NINA](https://nighttime-imaging.com/)，按现有部署方式将 `FMoraes.NINA.SitesPlugin.dll` 放入其插件目录，并核对望远镜连接。
- 核对 FTP 地址、数据目录和工作流使用的服务地址。

观测设备的安全联锁和人工确认仍由实际系统负责。没有连接设备时，本地运行只能验证相应的软件步骤。

## 接口与工作流

接口定义位于 `src/app/app2.py`，参数与返回格式以运行中的 API 文档为准：

| 任务 | 接口路径 |
| --- | --- |
| 查看或修改配置 | `/look_config`、`/modify_config` |
| 生成观测计划 | `/plan_observation` |
| 查询计划和日志 | `/get_oblist`、`/check_log` |
| 添加观测目标 | `/add_fine_oblist` |
| 传输计划与数据 | `/ftp_transfer` |
| 操作 NINA | `/manipulate_nina/{action}` |
| 暂现源处理与结果读取 | `/pipeline_transient`、`/get_filtered_csv_data` |
| 更新台站资料 | `/update_station` |

部分接口会修改配置、传输文件或操作设备，调用前需明确动作与参数。n8n 示例通过这些接口安排任务并读取日志；语言模型与服务地址应根据本机环境配置，模型能返回工具调用并不代表调用已经成功。

## TNS 辅助脚本

```bash
python Pachong.py
```

这个旧脚本读取公开 TNS 页面，可能随页面或登录方式变化而失效。它不控制望远镜，也不运行 [Explore 合成实验](../explore/README.md)。

引用实验结果时，应注明来自真实观测、历史日志还是合成环境。NGSS 的观测系统与本仓库公开的 Explore 指标表分别使用。
