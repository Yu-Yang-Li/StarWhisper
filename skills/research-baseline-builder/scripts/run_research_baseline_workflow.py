#!/usr/bin/env python
"""Run the research-baseline init flow and optionally execute the copied baseline.

This script is a thin workflow wrapper. It does not change model templates; it
only calls the existing init script, records progress, and updates status files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INIT_SCRIPT = SCRIPT_DIR / "init_research_baseline_workspace.py"


def clock() -> str:
    return datetime.now().strftime("%H:%M")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def progress(title: str, message: str) -> None:
    print(f"[{clock()} | {title}]")
    print(message)
    print()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def workspace_from_init_output(output: str) -> Path:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("init script did not print a workspace path")
    return Path(lines[-1])


def baseline_command(script_name: str, args: argparse.Namespace) -> list[str] | None:
    base = [sys.executable, str(Path("scripts") / script_name)]
    if script_name == "baseline_randomforest.py":
        cmd = base + ["--task", args.task]
        if args.csv:
            cmd += ["--csv", str(args.csv), "--target", args.target]
        return cmd
    if script_name == "baseline_xgboost_optuna.py":
        cmd = base + ["--task", args.task, "--n_trials", str(args.n_trials)]
        if args.csv:
            cmd += ["--csv", str(args.csv), "--target", args.target]
        return cmd
    if script_name == "baseline_gru.py":
        cmd = base + ["--epochs", str(args.epochs), "--seq_len", str(args.seq_len)]
        if args.csv:
            cmd += ["--csv", str(args.csv), "--value-column", args.value_column]
        return cmd
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="科研数据基线流程串联入口")
    parser.add_argument("topic", help="研究课题简述")
    parser.add_argument("--root", type=Path, help="输出根目录")
    parser.add_argument(
        "--template",
        choices=["rf", "xgb_optuna", "gru", "efficientnet", "none"],
        help="显式指定模板；不填时沿用 init 脚本自动路由。",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-baseline", action="store_true", help="初始化后运行一次复制出的 baseline。")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--csv", type=Path, help="表格/时序 CSV。表格任务需配 --target；时序任务需配 --value-column。")
    parser.add_argument("--target", help="表格 CSV 的目标列。")
    parser.add_argument("--value-column", help="时序 CSV 的数值列。")
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=12)
    args = parser.parse_args()

    if args.csv and args.template in {"rf", "xgb_optuna"} and not args.target:
        parser.error("--csv with tabular baseline requires --target")
    if args.csv and args.template == "gru" and not args.value_column:
        parser.error("--csv with GRU baseline requires --value-column")

    progress("任务接收", f"课题：{args.topic}；运行模式：{'初始化+基线' if args.run_baseline else '仅初始化'}。")
    init_cmd = [sys.executable, str(INIT_SCRIPT), args.topic]
    if args.root:
        init_cmd += ["--root", str(args.root)]
    if args.template:
        init_cmd += ["--template", args.template]
    if args.force:
        init_cmd.append("--force")

    init_result = run_command(init_cmd)
    print(init_result.stdout, end="")
    workspace = workspace_from_init_output(init_result.stdout)
    status_path = workspace / "workflow_status.json"
    status = read_json(status_path)
    status.update(
        {
            "orchestrator": "run_research_baseline_workflow.py",
            "orchestrator_updated_at": now(),
            "init_command": init_cmd,
            "init_returncode": init_result.returncode,
        }
    )

    if init_result.returncode != 0:
        status["status"] = "init_failed"
        status["init_output"] = init_result.stdout[-4000:]
        write_json(status_path, status)
        raise SystemExit(init_result.returncode)

    copied = status.get("copied_template")
    script_name = Path(copied).name if copied else None
    if not args.run_baseline:
        status["status"] = "initialized"
        status["next_step"] = "已完成工作区初始化；如需验证环境或真实数据，可用 --run-baseline 运行复制出的模板。"
        write_json(status_path, status)
        progress("流程暂停", f"已生成工作区：{workspace}；尚未运行模型，避免把 demo 当成真实结果。")
        return

    if not script_name:
        status["status"] = "baseline_skipped"
        status["baseline_skip_reason"] = "没有复制 baseline 脚本。"
        write_json(status_path, status)
        progress("基线跳过", "本次没有复制 baseline 脚本，因此只保留 SOP 和数据契约文件。")
        return

    cmd = baseline_command(script_name, args)
    if cmd is None:
        status["status"] = "baseline_skipped"
        status["baseline_skip_reason"] = "图像 EfficientNet 模板可能触发预训练权重下载，默认不由编排器自动运行。"
        write_json(status_path, status)
        progress("基线跳过", "已生成图像模板；自动运行可能下载权重，保留给用户确认真实图片目录后执行。")
        return

    if args.csv and script_name in {"baseline_randomforest.py", "baseline_xgboost_optuna.py"} and not args.target:
        status["status"] = "input_error"
        status["baseline_skip_reason"] = "--csv with tabular baseline requires --target"
        write_json(status_path, status)
        parser.error("--csv with tabular baseline requires --target")
    if args.csv and script_name == "baseline_gru.py" and not args.value_column:
        status["status"] = "input_error"
        status["baseline_skip_reason"] = "--csv with GRU baseline requires --value-column"
        write_json(status_path, status)
        parser.error("--csv with GRU baseline requires --value-column")

    progress(
        "基线运行",
        f"脚本：scripts/{script_name}；数据：{'用户 CSV' if args.csv else 'demo/合成数据'}；工作区：{workspace}",
    )
    baseline_result = run_command(cmd, cwd=workspace)
    print(baseline_result.stdout, end="")
    run_log = workspace / "workflow_run_log.txt"
    run_log.write_text(baseline_result.stdout, encoding="utf-8")
    summary_path = workspace / "baseline_summary.json"
    metrics_path = workspace / "metrics.json"
    status.update(
        {
            "status": "baseline_completed" if baseline_result.returncode == 0 else "baseline_failed",
            "baseline_updated_at": now(),
            "baseline_command": cmd,
            "baseline_returncode": baseline_result.returncode,
            "baseline_output_log": str(run_log.resolve()),
            "baseline_summary": str(summary_path.resolve()) if summary_path.exists() else None,
            "metrics": str(metrics_path.resolve()) if metrics_path.exists() else None,
            "data_mode": "user_csv" if args.csv else "demo",
        }
    )
    write_json(status_path, status)
    progress(
        "流程完成" if baseline_result.returncode == 0 else "流程中断",
        f"状态：{status['status']}；结果文件：{summary_path if summary_path.exists() else '未生成'}。",
    )
    raise SystemExit(baseline_result.returncode)


if __name__ == "__main__":
    main()
