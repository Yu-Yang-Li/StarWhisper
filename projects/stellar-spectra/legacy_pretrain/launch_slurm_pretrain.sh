#!/bin/bash
#SBATCH --gpus=8
module load miniconda/24.9.2 cuda/12.4
source activate lssttrans
# 8卡版本启动脚本（可通过 NPROC_PER_NODE 覆盖）
# 用法：
# 1) 先拿交互资源：salloc -p n76hb -N 1 --gpus=4 --ntasks=4 --cpus-per-task=16 --mem=0 -t 04:00:00
# 2) 然后在计算节点上：bash launch_n76hb_4gpus.sh
set -euo pipefail

# 强制使用硬编码绝对路径作为工作目录，避免在 SLURM 中落到 /var/spool
BASE_DIR="/data/home/scyb121/aolsst/spec"
cd "${BASE_DIR}"

######################## 基础环境（沿用你单卡脚本） ########################
# module load miniconda/24.9.2 cuda/12.4
# source activate lssttrans
# 注：已在 lssttrans 环境中，无需重复激活

# 可配置进程数（4卡4090）
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# 与 NCCL/内存/线程相关的环境（多卡优化配置）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export TOKENIZERS_PARALLELISM=true
export CSV_ENGINE=${CSV_ENGINE:-}       # PyArrow不支持chunksize，使用默认C引擎
export ARROW_NUM_THREADS=${ARROW_NUM_THREADS:-32}  # Arrow 解析线程数（一般与 OMP_NUM_THREADS 一致）

# 数据输入策略（超大数据默认走在线流式训练）
export STREAMING_MODE=${STREAMING_MODE:-1}        # 0: 内存Dataset  1: 流式IterableDataset
# 在线分片场景建议将块大小降低到40万，减少首批等待
export CSV_CHUNK_SIZE=${CSV_CHUNK_SIZE:-200000}
export SHARD_WAIT_TIMEOUT_SEC=${SHARD_WAIT_TIMEOUT_SEC:-72000}
export ASSUME_SORTED=${ASSUME_SORTED:-1}          # 若CSV已全局排序，开启可跳过块内排序
# 仅在STREAMING_MODE=1时有效：IterableDataset无法len，给个估计步数避免学习率异常
# 仅在 IterableDataset 无法计算长度时使用该估计值
# 如 batch size 更改，请按比例调整该值，保证 LR 调度稳定
# 参考步数估计（以全局batch=128为基准）
export ESTIMATED_STEPS_PER_EPOCH_BASE=${ESTIMATED_STEPS_PER_EPOCH_BASE:-17083}
export ESTIMATED_STEPS_PER_EPOCH=${ESTIMATED_STEPS_PER_EPOCH:-${ESTIMATED_STEPS_PER_EPOCH_BASE}}
# 流式多worker与预取，提升各rank首样本就绪速度
export STREAM_WORKERS=${STREAM_WORKERS:-1}
export STREAM_PREFETCH=${STREAM_PREFETCH:-2}
export STREAMING_MODE=${STREAMING_MODE:-1}   # 默认启用流式读取，加速验证/训练IO
export STREAM_PIN_MEMORY=${STREAM_PIN_MEMORY:-1}
export STREAM_PERSISTENT=${STREAM_PERSISTENT:-1}

# # NCCL 参数（4卡单机优化配置）
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_TREE_THRESHOLD=0
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-4}
# export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-2}
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_MIN_NRINGS=8
# export NCCL_ALGO=Tree,Ring

# NCCL 参数（4卡单机优化配置）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_BLOCKING_WAIT  # 避免与 ASYNC 冲突
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-4}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-2}
export NCCL_MIN_NRINGS=8
export NCCL_ALGO=Tree,Ring
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=0   # 先禁用 GDR，确认稳定后再放开
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-300}




# 在线分片与哈希划分（不落盘，在线筛分，适配800G超大数据）
export ON_THE_FLY_SHARDING=${ON_THE_FLY_SHARDING:-1}
# 使用独立的 train/val 文件：关闭单CSV内哈希划分
export HASH_SPLIT_ENABLE=${HASH_SPLIT_ENABLE:-0}
export HASH_SPLIT_BASE=${HASH_SPLIT_BASE:-10}
export HASH_SPLIT_TRAIN_THRESHOLD=${HASH_SPLIT_TRAIN_THRESHOLD:-9}
export SINGLE_CSV_PATH=${SINGLE_CSV_PATH:-}  # 仅当你想用单CSV时设置此项并开启 HASH_SPLIT_ENABLE=1

# 可覆盖的训练/验证CSV路径（使用你提供的绝对路径，便于直接运行）
export DATA_ROOT=${DATA_ROOT:-"/data/home/scyb121/aolsst/spec/pretrain_data"}
export TRAIN_CSV=${TRAIN_CSV:-"${DATA_ROOT}/spectrum_tokenized_train.csv"}
export VAL_CSV=${VAL_CSV:-"${DATA_ROOT}/spectrum_tokenized_val.csv"}
export VAL_SUBSET_CSV=${VAL_SUBSET_CSV:-"${DATA_ROOT}/spectrum_tokenized_val_subset.csv"}  # 步级验证用子集

# 步数检查点设置（只跑1个epoch时，建议用 SAVES_PER_EPOCH 控制保存次数）
export ENABLE_STEP_CHECKPOINTS=${ENABLE_STEP_CHECKPOINTS:-1}   # 开启步数检查点
export STEP_CHECKPOINT_INTERVAL=${STEP_CHECKPOINT_INTERVAL:-3000} # 每3000步保存
export KEEP_N_CHECKPOINTS=${KEEP_N_CHECKPOINTS:-20}     # 保留最近6个步数检查点
export KEEP_K_BEST=${KEEP_K_BEST:-3}                   # 仅保留最好的3个best
export KEEP_K_BEST_RMSE=${KEEP_K_BEST_RMSE:-3}         # 仅保留RMSE最好的3个best

# 步级验证策略（默认：每200步对1500子集做全量评估，确保基准一致）
export VAL_FREQUENCY=${VAL_FREQUENCY:-200}
export VAL_RMSE_BATCHES=${VAL_RMSE_BATCHES:-4}   # 仅在轻量模式下生效
export VAL_STEP_USE_FULL=${VAL_STEP_USE_FULL:-1} # 1: 每次遍历子集全部批次；0: 仅取前K批（更快）

# Epoch 检查点设置（配合脚本中的 config 默认值）
export EPOCH_CHECKPOINT_INTERVAL=${EPOCH_CHECKPOINT_INTERVAL:-1}        # 每个epoch保存
export KEEP_LAST_EPOCH_CHECKPOINTS=${KEEP_LAST_EPOCH_CHECKPOINTS:-5}    # 最多保留5个epoch检查点

# 词表路径（与你的 Python 脚本一致）
export VOCAB_PATH=${VOCAB_PATH:-"/data/home/scyb121/aolsst/spec/pretrain_data/vocabulary.csv"}
# 特殊token映射：按词表 `<BOS>=0 <EOS>=1 <SEP>=2`；mask 统一用 <SEP> 作为占位
export MASK_TOKEN_ID=${MASK_TOKEN_ID:-2}

# 分片目录使用项目默认位置（不建立软链接，不创建目录）

# 打开更高的文件句柄上限，降低IO阻塞概率
ulimit -n 1048576 2>/dev/null || true

# 正式训练模式（默认关闭基准测试，直接开始训练）
export IO_BENCHMARK=${IO_BENCHMARK:-0}
export IO_BENCHMARK_BATCHES=${IO_BENCHMARK_BATCHES:-50}

# 训练批大小与初始微批（避免首批OOM反复回退）
export BATCH_SIZE=${BATCH_SIZE:-128}
export INIT_MICRO_BATCH=${INIT_MICRO_BATCH:-8}

# 按全局batch自动缩放每epoch步数（基于参考：REF_BS=128 => 17083 步）
export AUTO_SCALE_EST_STEPS=${AUTO_SCALE_EST_STEPS:-1}
if [ "${AUTO_SCALE_EST_STEPS}" = "1" ]; then
  REF_BS=${REF_BS:-128}
  if [ "${BATCH_SIZE}" -gt 0 ]; then
    ESTIMATED_STEPS_PER_EPOCH=$(( ESTIMATED_STEPS_PER_EPOCH_BASE * REF_BS / BATCH_SIZE ))
    export ESTIMATED_STEPS_PER_EPOCH
  fi
fi

# 如你的代码已支持 VOCAB_PATH 环境变量，可在此启用
# export VOCAB_PATH=/abs/path/to/vocabulary.csv

LOG_DIR="${BASE_DIR}/logs"
OUT_DIR_PRIMARY="${BASE_DIR}/output"
OUT_DIR_LEGACY="${BASE_DIR}/pth"   # 兼容之前存放 checkpoint 的目录
mkdir -p "${LOG_DIR}" "${OUT_DIR_PRIMARY}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUT_DIR_PRIMARY}}"

# 强制从指定步数的 checkpoint 续训（适用于将数据分为上下半段时衔接）
export FORCE_RESUME_STEP=${FORCE_RESUME_STEP:-1}
export FORCE_RESUME_STEP_NUMBER=${FORCE_RESUME_STEP_NUMBER:-18000}

# 自动选择断点进行续训（优先步数检查点），无需额外传参
if [ -z "${RESUME_FROM:-}" ]; then
  # 依次在以下目录中查找：$OUTPUT_DIR、output/、pth/
  try_dirs=("${OUTPUT_DIR}" "${OUT_DIR_PRIMARY}" "${OUT_DIR_LEGACY}")
  # 0) 若启用强制步数续训且对应文件存在，则直接使用该文件
  if [ "${FORCE_RESUME_STEP}" = "1" ]; then
    for D in "${try_dirs[@]}"; do
      [ -d "${D}" ] || continue
      CAND="${D}/checkpoint_step_${FORCE_RESUME_STEP_NUMBER}.pth"
      if [ -f "${CAND}" ]; then
        RESUME_FROM="${CAND}"
        break
      fi
    done
  fi
  # 1) 优先使用 last_checkpoint.pth（包含优化器与调度器）
  if [ -z "${RESUME_FROM:-}" ]; then
    for D in "${try_dirs[@]}"; do
      [ -n "${D}" ] || continue
      if [ -f "${D}/last_checkpoint.pth" ]; then
        RESUME_FROM="${D}/last_checkpoint.pth"
        break
      fi
    done
  fi
  # 2) 否则在所有目录中寻找步数最大的 checkpoint_step_*.pth
  if [ -z "${RESUME_FROM:-}" ]; then
    best_step=0
    best_file=""
    for D in "${try_dirs[@]}"; do
      [ -d "${D}" ] || continue
      for f in "${D}"/checkpoint_step_*.pth; do
        [ -e "${f}" ] || continue
        bn="$(basename -- "${f}")"
        step="${bn#checkpoint_step_}"
        step="${step%.pth}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && [ "${step}" -gt "${best_step}" ]; then
          best_step="${step}"
          best_file="${f}"
        fi
      done
    done
    if [ -n "${best_file}" ]; then
      RESUME_FROM="${best_file}"
    fi
  fi
  # 3) 最后兜底：last_step_model.pth（若包含完整状态亦可续训，否则仅能载入权重）
  if [ -z "${RESUME_FROM:-}" ]; then
    for D in "${try_dirs[@]}"; do
      [ -n "${D}" ] || continue
      if [ -f "${D}/last_step_model.pth" ]; then
        RESUME_FROM="${D}/last_step_model.pth"
        break
      fi
    done
  fi
fi
export RESUME_FROM
echo "[Resume] RESUME_FROM=${RESUME_FROM:-<none>}"

######################## 预检查（避免空跑） ########################
python - <<'PY'
import os, torch, sys
n = torch.cuda.device_count()
need = int(os.environ.get("NPROC_PER_NODE", "4"))
print(f"[Check] PyTorch={torch.__version__}  CUDA GPUs={n}  need_procs={need}")
sys.exit(0 if n>=need else 1)
PY

######################## 启动多卡（DDP） ########################
# 若设置 REPO_ROOT 为克隆的本仓库路径，则使用 scripts/pretrain.py + src/spectral_lm（推荐开源布局）
# 未设置时沿用 BASE_DIR 下旧文件名（需与同目录的 model_architecture.py 配套）
REPO_ROOT="${REPO_ROOT:-}"
if [ -n "${REPO_ROOT}" ] && [ -f "${REPO_ROOT}/scripts/pretrain.py" ]; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  PYFILE="${REPO_ROOT}/scripts/pretrain.py"
else
  PYFILE="${BASE_DIR}/pretrain_script_legacy.py"
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
# 若默认端口被占用，自动查找可用端口（29500-29999）；依赖 ss，仅 Linux 常见环境可用
if ss -Htan | awk '{print $4}' | grep -q ":${MASTER_PORT}$"; then
  for p in $(seq 29500 29999); do
    if ! ss -Htan | awk '{print $4}' | grep -q ":${p}$"; then
      MASTER_PORT=$p
      break
    fi
  done
fi

echo "[Run] torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port ${MASTER_PORT} ${PYFILE}"
# 防止集体通信超时过早触发，给较大的超时时间（分钟）
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
mkdir -p "${LOG_DIR}"
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port "${MASTER_PORT}" "${PYFILE}" 2>&1 | tee -a "${LOG_DIR}/run_${NPROC_PER_NODE}g_$(date +%Y%m%d_%H%M%S).log"
