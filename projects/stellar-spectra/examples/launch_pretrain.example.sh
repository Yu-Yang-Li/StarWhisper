#!/usr/bin/env bash
# 预训练启动示例（多卡 DDP）。请按集群环境修改 module / conda，并设置数据路径。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# 数据与词表（改为你的路径）
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/pretrain_tokenized}"
export TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/spectrum_tokenized_train.csv}"
export VAL_CSV="${VAL_CSV:-${DATA_ROOT}/spectrum_tokenized_val.csv}"
# 步级验证用小验证集；若不存在则预训练脚本会回退用 VAL_CSV（见 scripts/pretrain.py）
export VAL_SUBSET_CSV="${VAL_SUBSET_CSV:-${DATA_ROOT}/spectrum_tokenized_val_subset.csv}"
export VOCAB_PATH="${VOCAB_PATH:-${REPO_ROOT}/vocab/vocabulary.csv}"
export MASK_TOKEN_ID="${MASK_TOKEN_ID:-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/pretrain}"

mkdir -p "${OUTPUT_DIR}"

# 多机多卡时需设置 MASTER_ADDR（本机单机可省略）
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT:-29500}" \
  "${REPO_ROOT}/scripts/pretrain.py"
