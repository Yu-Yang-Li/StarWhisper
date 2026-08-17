#!/usr/bin/env python
"""Create a minimal research baseline workspace.

新增: --template 参数，可选 {rf, xgb_optuna, gru, efficientnet, none}
- 指定模板时，会把 templates/baseline_<type>.py 复制到生成目录的 scripts/ 下
- baseline_plan.md 中追加"推荐基线模型"章节；none 时给出 4 类决策表
- 若 topic 命中关键词（image/视觉/图像/显微/天文/timeseries/时序/序列/forecast/
  tabular/表格/结构化），自动推荐对应模板；显式 --template 优先级最高。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = SKILL_DIR / "templates"

TEMPLATE_FILES = {
    "rf": "baseline_randomforest.py",
    "xgb_optuna": "baseline_xgboost_optuna.py",
    "gru": "baseline_gru.py",
    "efficientnet": "baseline_efficientnet.py",
}

TEMPLATE_RECS = {
    "rf": "- **基线模型**：RandomForest（表格数据快速强基线，可解释性好），见 `scripts/baseline_randomforest.py`\n"
          "- 依赖：scikit-learn, pandas, numpy, joblib\n",
    "xgb_optuna": "- **基线模型**：XGBoost + Optuna（表格 SOTA 强基线 + 超参搜索），见 `scripts/baseline_xgboost_optuna.py`\n"
                  "- 依赖：xgboost>=2.0, optuna, scikit-learn, pandas, numpy, joblib\n",
    "gru": "- **基线模型**：GRU（时序预测基线），见 `scripts/baseline_gru.py`\n"
           "- 依赖：torch, numpy, pandas, scikit-learn\n",
    "efficientnet": "- **基线模型**：EfficientNet-B0（图像分类迁移学习基线），见 `scripts/baseline_efficientnet.py`\n"
                    "- 依赖：torch, torchvision, pillow\n",
}


FILES = {
    "problem_definition.md": "# Problem Definition\n\n## Scientific question\n\n## Research goal\n\n## Input data\n\n## Data description / field meaning\n\n## Expected output\n\n## Unit of analysis\n\n## Data question\n\n## Recommended framework\n\n## Outcome / label / effect\n\n## Features known before output\n\n## Leakage risks\n\n## Missing fields\n",
    "eda_plan.md": "# EDA Plan\n\n- Label/outcome distribution\n- Missingness\n- Feature distributions\n- Group/time/batch/source balance\n- Leakage checks\n",
    "preprocess_plan.md": "# Preprocess Plan\n\n- Cleaning and units\n- Missing values\n- Outliers\n- Encoding\n- Scaling\n- Split protocol\n",
    "baseline_plan.md": "# Baseline Plan\n\n## Sanity baseline\n\n## Interpretable baseline\n\n## Strong classical baseline\n\n## Stronger model only if justified\n",
    "train_eval_plan.md": "# Train and Evaluation Plan\n\n## Split\n\n## Primary metric\n\n## Secondary metrics\n\n## Uncertainty\n\n## Subgroup/error analysis\n",
    "baseline_report.md": "# Baseline Report\n\n## Original scientific goal\n\n## Data task\n\n## Dataset summary\n\n## Framework used\n\n## Models or analyses tested\n\n## Results\n\n## Error analysis\n\n## Scientific interpretation\n\n## Does this answer the goal?\n\n## Next step\n",
}

SCHEMA_HEADER = [
    "field", "role", "type", "required", "known_before_outcome", "notes",
]


DECISION_TABLE = """
## 推荐基线模型

| 问题类型 | 数据形态 | 快速基线 | 强基线 |
|---------|---------|---------|--------|
| 表格/结构化 | DataFrame (n_samples, n_features) | **RandomForest** | **XGBoost + Optuna** |
| 时序/序列预测 | 1D/多维 array (T,) / (T, F) | 历史均值/ARIMA | **GRU** |
| 图像/视觉 | 图片文件夹 (train/val/class/img.jpg) | 预训练特征+线性头 | **EfficientNet-B0** |

> 先跑"快速基线"作为标杆，确认数据/特征/评估无误后再上"强基线"。
"""


def topic_slug(text: str) -> str:
    if not text:
        return "research-baseline"
    head = str(text).strip()[:20]
    head = re.sub(r"\s+", "-", head)
    head = re.sub(r"[^\w\u4e00-\u9fff\-]+", "", head, flags=re.UNICODE)
    head = re.sub(r"-+", "-", head).strip("-")
    return head or "research-baseline"


def default_root(topic: str) -> Path:
    return Path(".") / "research-baseline" / topic_slug(topic)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clock() -> str:
    return datetime.now().strftime("%H:%M")


def progress(title: str, message: str) -> None:
    print(f"[{clock()} | {title}]")
    print(message)
    print()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_template(topic: str) -> dict:
    """基于 topic 关键词推荐模板，并保留可审计路由理由。"""
    t = topic.lower()
    groups = [
        (
            "efficientnet",
            ["image", "图像", "视觉", "显微", "天文", "影像"],
            "图像/视觉类科研数据，优先复制 EfficientNet 迁移学习模板。",
        ),
        (
            "xgb_optuna",
            ["tabular", "表格", "结构化"],
            "表格/结构化任务，优先复制 XGBoost+Optuna 强基线模板。",
        ),
        (
            "gru",
            ["time series", "timeseries", "时序", "序列", "forecast"],
            "明确时序/序列任务，优先复制 GRU 时序模板。",
        ),
    ]
    for template_key, keywords, reason in groups:
        matched = [keyword for keyword in keywords if keyword in t]
        if matched:
            return {
                "selected_template": template_key,
                "matched_keywords": matched,
                "reason": reason,
                "fallback_used": False,
            }
    return {
        "selected_template": "rf",
        "matched_keywords": [],
        "reason": "未命中明确模态关键词，使用 RandomForest 作为表格通用快速基线。",
        "fallback_used": True,
    }


def copy_template(template_key: str, scripts_dir: Path) -> str | None:
    if template_key not in TEMPLATE_FILES:
        return None
    src = TEMPLATES_DIR / TEMPLATE_FILES[template_key]
    if not src.exists():
        print(f"[warn] 模板文件不存在: {src}")
        return None
    dst = scripts_dir / TEMPLATE_FILES[template_key]
    shutil.copy2(src, dst)
    print(f"[模板] 复制 {src.name} -> {dst}")
    return TEMPLATE_FILES[template_key]


def append_baseline_section(baseline_plan: Path, template_key: str, auto: bool):
    text = baseline_plan.read_text(encoding="utf-8") if baseline_plan.exists() else "# Baseline Plan\n"
    if "## 推荐基线模型" in text:
        return
    addition = "\n\n" + DECISION_TABLE
    if template_key != "none":
        tag = " (自动识别)" if auto else ""
        addition += f"\n**本次选用模板{tag}**：\n\n{TEMPLATE_RECS[template_key]}"
    baseline_plan.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 research baseline 工作空间")
    parser.add_argument("topic", help="研究课题简述，用于生成目录名")
    parser.add_argument("--root", default=None,
                        help="输出根目录，默认为 ./research-baseline/<topic-slug>")
    parser.add_argument("--template", choices=list(TEMPLATE_FILES.keys()) + ["none"],
                        default=None,
                        help="选择可运行基线模板：rf(表格RandomForest), xgb_optuna(表格XGBoost+Optuna), "
                             "gru(时序GRU), efficientnet(图像EfficientNet), none(不复制，仅放决策表)。"
                             "默认根据 topic 关键词自动识别。")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # 决定模板（显式 > 自动识别）
    route = None
    if args.template is None:
        route = detect_template(args.topic)
        template_key = route["selected_template"]
        selection_mode = "auto"
    else:
        template_key = args.template
        route = {
            "selected_template": template_key,
            "matched_keywords": [],
            "reason": "用户通过 --template 显式指定模板。",
            "fallback_used": False,
        }
        selection_mode = "explicit"

    if args.root:
        out = Path(args.root) / topic_slug(args.topic)
    else:
        out = default_root(args.topic)

    progress(
        "任务路由",
        f"课题：{args.topic}；模板：{template_key}；依据：{route['reason']}",
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    scripts_dir = out / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    generated_files = []
    for name, content in FILES.items():
        path = out / name
        if args.force or not path.exists():
            path.write_text(content, encoding="utf-8")
        generated_files.append(name)

    schema = out / "data_schema.csv"
    if args.force or not schema.exists():
        with schema.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(SCHEMA_HEADER)
    generated_files.append("data_schema.csv")

    # 复制模板
    copied_script = None
    if template_key != "none":
        copied_script = copy_template(template_key, scripts_dir)

    # 更新 baseline_plan.md
    append_baseline_section(out / "baseline_plan.md", template_key, selection_mode == "auto")

    routing_decision = {
        "schema": "research_baseline_routing_decision_v1",
        "created_at": now(),
        "topic": args.topic,
        "selection_mode": selection_mode,
        "selected_template": template_key,
        "matched_keywords": route["matched_keywords"],
        "reason": route["reason"],
        "fallback_used": route["fallback_used"],
        "copied_script": f"scripts/{copied_script}" if copied_script else None,
    }
    write_json(out / "routing_decision.json", routing_decision)

    workflow_status = {
        "schema": "research_baseline_workflow_status_v1",
        "created_at": now(),
        "status": "initialized",
        "topic": args.topic,
        "workspace": str(out.resolve()),
        "data_mode": "no_user_data",
        "generated_files": generated_files,
        "copied_template": f"scripts/{copied_script}" if copied_script else None,
        "routing_decision": str((out / "routing_decision.json").resolve()),
        "next_step": "填写 data_schema / 替换模板数据加载；如只验证环境，可运行模板 demo，但不要把 demo 指标当成真实数据结果。",
    }
    write_json(out / "workflow_status.json", workflow_status)

    progress(
        "工作区",
        f"已创建 {out.resolve()}；生成 {len(generated_files)} 个 SOP/Schema 文件；模板脚本：{copied_script or '未复制'}。",
    )

    print(out.resolve())


if __name__ == "__main__":
    main()
