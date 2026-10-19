#!/usr/bin/env bash
# 微调启动示例。通过环境变量区分信噪比实验；与 scripts/finetune.py 配合使用。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# 加载某套 SNR 的默认路径（可改为 source examples/env_finetune_snr_25_30.example.sh）
if [ -n "${FINETUNE_ENV_FILE:-}" ] && [ -f "${FINETUNE_ENV_FILE}" ]; then
  # shellcheck source=/dev/null
  source "${FINETUNE_ENV_FILE}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

: "${TRAIN_CSV:?请设置 TRAIN_CSV}"
: "${VAL_CSV:?请设置 VAL_CSV}"
: "${VOCAB_PATH:?请设置 VOCAB_PATH}"
: "${PRETRAIN_CKPT_PATH:?请设置 PRETRAIN_CKPT_PATH}"

# 以下有合理默认（相对 REPO_ROOT）；生产环境建议通过 env 文件覆盖
export STEP_EVAL_CSV="${STEP_EVAL_CSV:-${REPO_ROOT}/data/finetune_tokenized/spectrum_tokenized_val_first1500.csv}"
export CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/output/finetune_ckpts}"
export LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/finetune_$(date +%Y%m%d_%H%M%S).log}"
# 参数标准化：可选；不设置时由训练集流式估计（PARAM_STATS_SAMPLES 控制采样条数）
# export PARAM_STATS_JSON="${REPO_ROOT}/data/param_stats.json"
export PARAM_STATS_SAMPLES="${PARAM_STATS_SAMPLES:-256}"
# 与集群脚本一致：大数据用流式；若数据很小可改为 1 做内存加载
export PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-0}"
export STREAMING_MODE="${STREAMING_MODE:-1}"

mkdir -p "$(dirname "${LOG_PATH}")" "${CKPT_DIR}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/scripts/finetune.py"
