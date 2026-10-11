# 基线模型模板与依赖

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## 基线模型选择决策表

明确输入、输出和样本单位后，按数据类型选择一个简单基线。运行时间取决于数据规模和环境；先检查数据与评估方法，再决定是否增加模型复杂度。

| 问题类型 | 数据形态 | 初始基线 | 后续候选 | 代码模板 |
|---------|---------|---------|--------|---------|
| 表格/结构化 (分类/回归) | DataFrame `(n_samples, n_features)` | **RandomForest** | **XGBoost + Optuna**（需限定搜索范围） | `baseline_randomforest.py` → `baseline_xgboost_optuna.py` |
| 时序/序列预测 | 1D/多维 array `(T,)` / `(T, F)` | 历史均值 / 季节性 Naive | **GRU** (PyTorch) | `baseline_gru.py` |
| 图像/视觉分类 | 图片文件夹 `train/<class>/xxx.jpg` | 预训练特征 + 线性头 | **EfficientNet-B0** 迁移学习（冻结→微调） | `baseline_efficientnet.py` |

未指定 `--template` 时，脚本按关键词选择：
- 包含 `image/图像/视觉/显微/天文/影像` → 默认 `efficientnet`
- 包含 `tabular/表格/结构化` → 默认 `xgb_optuna`
- 包含 `time series/timeseries/时序/序列/forecast` → 默认 `gru`
- 其他 → 默认 `rf`（表格通用基线）

关键词选择只是提示，不能判断数据的实际形态。例如“天文”也会匹配图像模板；处理光变特征表或观测日志时，应显式指定 `--template rf`、`xgb_optuna` 或 `gru`。

## 代码模板

`templates/` 中有 4 个可独立运行的脚本，文件开头说明数据格式与依赖。表格和时序模板支持 CSV，图像模板读取分类目录。没有用户数据时可用自带或合成数据检查环境；这些结果不能作为课题的实验结果。

| 模板文件 | 模型 | 框架 | 依赖 | 适用场景 |
|---------|------|------|------|---------|
| `baseline_randomforest.py` | RandomForest 分类/回归 | scikit-learn | `scikit-learn, pandas, numpy, joblib` | 表格数据快速基线、可解释性、特征重要性 |
| `baseline_xgboost_optuna.py` | XGBoost + Optuna 超参搜索 | xgboost + optuna | `xgboost>=2.0, optuna, scikit-learn, pandas, numpy, joblib` | 表格分类或回归，需要调参时使用 |
| `baseline_gru.py` | GRU 时序预测 | PyTorch | `torch, numpy, pandas, scikit-learn` | 单/多变量时序、传感器/金融/气象/实验数据 |
| `baseline_efficientnet.py` | EfficientNet-B0 迁移学习 | PyTorch + torchvision | `torch, torchvision, pillow` | 图像分类、科学图像（显微/天文/医学）、小样本 |

> 依赖可按需安装；没装 torch 的话 rf / xgb_optuna 两个表格模板依然可用，互不依赖。

### 使用示例

以下初始化命令在本技能目录中运行，每次只复制所选的一个模型模板：

```bash
python scripts/init_research_baseline_workspace.py "光变特征分类" --template rf --root ./lightcurve-baseline
python scripts/init_research_baseline_workspace.py "全天相机图像分类" --template efficientnet
python scripts/init_research_baseline_workspace.py "巡天天气时序预测" --template gru
python scripts/init_research_baseline_workspace.py "超新星候选筛选" --template none
```

以第一个工作目录为例，进入后运行复制出的 RandomForest 模板：

```bash
cd ./lightcurve-baseline/光变特征分类
python scripts/baseline_randomforest.py --help
python scripts/baseline_randomforest.py  # 自带 demo，仅检查环境
python scripts/baseline_randomforest.py --csv ./data.csv --target label
```

`data.csv` 和 `label` 要替换成实际文件与目标列。结果写到该工作目录下。选择了 XGBoost 或 GRU 模板的目录可分别运行：

```bash
python scripts/baseline_xgboost_optuna.py --csv ./data.csv --target label --n_trials 20
python scripts/baseline_gru.py --csv ./series.csv --value-column signal --epochs 10
```

也可回到本技能目录，由工作流脚本完成初始化和一次演示运行：

```bash
python scripts/run_research_baseline_workflow.py "光变特征分类" --template rf --root ./lightcurve-demo --run-baseline
```
