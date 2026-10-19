#!/usr/bin/env bash
# 较低信噪比微调（约 SNR 7–9）。典型做法：从 SNR 9–11 阶段最优 checkpoint 初始化。

export VOCAB_PATH="${VOCAB_PATH:-/path/to/repo/vocab/vocabulary.csv}"
export PRETRAIN_CKPT_PATH="${PRETRAIN_CKPT_PATH:-/path/to/finetune_snr9_11/ckpts/checkpoint_step_XXXX.pth}"
export TRAIN_CSV="${TRAIN_CSV:-/path/to/finetune_snr7_9/pretrain_data/spectrum_tokenized_train.csv}"
export VAL_CSV="${VAL_CSV:-/path/to/finetune_snr7_9/pretrain_data/spectrum_tokenized_val.csv}"
export STEP_EVAL_CSV="${STEP_EVAL_CSV:-/path/to/finetune_snr7_9/pretrain_data/spectrum_tokenized_val_first1500.csv}"
export CKPT_DIR="${CKPT_DIR:-/path/to/finetune_snr7_9/ckpts}"
export PARAM_STATS_JSON="${PARAM_STATS_JSON:-/path/to/param_stats.json}"
export RESUME_FROM="${RESUME_FROM:-${PRETRAIN_CKPT_PATH}}"
export PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-0}"
