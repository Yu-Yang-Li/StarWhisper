# NGSS · StarWhisper Telescope

近邻星系巡天（NGSS）的观测自动化代码，对应论文 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。程序负责制定观测计划、管理目标、连接 NINA，以及传输和处理观测数据。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [配置工具说明](../../skills/starwhisper-night-plan/SKILL.md)

下载后可以直接检查附带的观测配置。完整服务还需要补齐计算模块、台站数据和设备通信设置，不能仅安装依赖后就开始观测。

## 先检查配置

需要 Python 3.10 或更新版本，只依赖标准库。从本目录（`projects/ngss/`）运行：

```bash
python ../../skills/starwhisper-night-plan/scripts/plan_night.py check-config
python ../../skills/starwhisper-night-plan/scripts/plan_night.py budget
```

工具只读取本目录的 `observe_config.json`，不启动服务、不连接设备。当前配置为每目标 3 次、每次 120 秒曝光，另加 1 分钟指向与读出时间，共 420 秒；6.5 小时窗口的简单预算可容纳 55 个目标。实际计划还需考虑目标可见性、天气和设备条件。

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

## 完整服务还需要什么

现有源码保留了台站部署时的目录和通信约定。搭建自己的系统前，需要准备：

- **计算模块与本地数据。** `PlanObservation3.py` 引用的 `src/module/observable_calculator.py` 未收录；`data/` 下的台站表、`50_bright_03.csv` 星表、按日期保存的计划与配置也未附带。代码使用的 `de440s.bsp`、`finals2000A.all` 及其路径需另行核对。
- **设备与通信。** 安装 [NINA](https://nighttime-imaging.com/) 及 `FMoraes.NINA.SitesPlugin.dll`，配置望远镜、FTP 和 MQTT。`src/module/UdpConnect.py` 中的通信初始化需要按自己的网络补齐，并核对 `topic.yaml`。
- **数据处理环境。** 按 [X-OPSTEP](https://github.com/Shinehope-Chen/X-OPSTEP/blob/master/README.md) 说明准备 astrometry.net、ASTAP、Scamp、SWarp、HOTPANTS，以及观测图像与模板。

源码中的 `/home/pod/...` 是原部署位置，换机器时需逐项调整。根目录的 `observe_config.json` 用于查看配置格式；服务还会读取 `data/` 中的日期配置，不能只修改根目录这一份。

### 准备好后怎样启动

以下是服务入口，需先补齐上述模块、数据和配置。从本目录运行：

```bash
conda env create -f observe.yml
conda activate observe
uvicorn src.app.app2:app --reload
```

`observe.yml` 的环境名为 `observe`，其中保留了 Linux 平台的包构建版本；其他平台需要调整依赖。服务默认使用端口 8000，启动后可查看[本地 API 文档](http://127.0.0.1:8000/docs)。`--reload` 用于开发调试；服务启动成功仍需分别验证设备连接和观测流程。

## 观测配置

配置使用平铺字段。仓库中的示例为：

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

检查自己的配置时，将全局参数 `--config /path/to/observe_config.json` 放在 `check-config` 或 `budget` 前。目标表检查和参数说明见 [`starwhisper-night-plan`](../../skills/starwhisper-night-plan/SKILL.md)。

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

部分接口会修改配置、传输文件或操作设备，调用前需明确动作与参数，并遵守台站权限和安全联锁。n8n 示例通过这些接口安排任务并读取日志；语言模型与服务地址应根据本机环境配置，模型返回工具调用后还需检查服务的实际执行结果。

## TNS 辅助脚本

[`Pachong.py`](Pachong.py) 是早期读取公开 TNS 页面的辅助脚本，依赖当时的页面结构，可能随页面或登录方式变化而失效。它不控制望远镜，也不运行 [Explore 合成实验](../explore/README.md)。

引用实验结果时，应注明来自真实观测、历史日志还是合成环境。NGSS 的观测系统与本仓库公开的 Explore 指标表分别使用。
