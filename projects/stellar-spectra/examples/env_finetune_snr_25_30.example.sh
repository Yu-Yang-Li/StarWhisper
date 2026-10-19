#!/usr/bin/env bash
# 高信噪比微调（约 SNR 25–30）路径示例。复制为 env 文件后修改绝对路径，再:
#   export FINETUNE_ENV_FILE=/path/to/your/env.sh
#   bash examples/launch_finetune.example.sh

export VOCAB_PATH="${VOCAB_PATH:-/path/to/repo/vocab/vocabulary.csv}"
export PRETRAIN_CKPT_PATH="${PRETRAIN_CKPT_PATH:-/path/to/pretrain_or_prev_stage/ckpts/checkpoint_step_XXXX.pth}"
export TRAIN_CSV="${TRAIN_CSV:-/path/to/finetune_snr25_30/pretrain_data/spectrum_tokenized_train.csv}"
export VAL_CSV="${VAL_CSV:-/path/to/finetune_snr25_30/pretrain_data/spectrum_tokenized_val.csv}"
export STEP_EVAL_CSV="${STEP_EVAL_CSV:-/path/to/finetune_snr25_30/pretrain_data/spectrum_tokenized_val_first1500.csv}"
export CKPT_DIR="${CKPT_DIR:-/path/to/finetune_snr25_30/ckpts}"
export PARAM_STATS_JSON="${PARAM_STATS_JSON:-/path/to/param_stats.json}"
export LOG_PATH="${LOG_PATH:-/path/to/finetune_snr25_30/log/finetune.log}"
export PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-0}"
