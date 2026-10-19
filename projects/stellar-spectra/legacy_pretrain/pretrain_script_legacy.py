"""
简化但高效的全流程光谱扩散模型训练
- 预训练阶段：专注于flux RMSE监控
- 微调阶段：专注于参数预测RMSE监控
- 实时训练进度可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
from model_architecture import SpectrumDiffusionModel
from tqdm import tqdm
import random
import time
import math
from contextlib import nullcontext
import warnings
warnings.filterwarnings('ignore')
import hashlib

# 保持默认数值路径与步数行为（不主动修改TF32/SDPA/benchmark等全局开关）

class FocalLoss(nn.Module):
    """Focal Loss实现 - 解决类别不平衡问题（支持忽略特殊token）"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_token_ids: set[int] | None = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_token_ids: set[int] = set(ignore_token_ids or [])

    def set_ignore_token_ids(self, ids: set[int] | list[int] | tuple[int, ...]):
        self.ignore_token_ids = set(ids or [])
    
    def forward(self, inputs, targets):
        # inputs: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        
        # 重塑为2D
        vocab = inputs.size(-1)
        inputs = inputs.view(-1, vocab)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        # 构造有效mask（忽略 PAD/BOS/EOS 等）
        if self.ignore_token_ids:
            ignore_mask = torch.zeros_like(targets, dtype=torch.bool)
            for tid in self.ignore_token_ids:
                ignore_mask |= (targets == tid)
            valid_mask = ~ignore_mask
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        if valid_mask.any():
            # 仅对有效位置计算CE
            ce_loss_all = F.cross_entropy(inputs, targets, reduction='none')
            ce_loss = ce_loss_all[valid_mask]
            # 计算概率
            pt = torch.exp(-ce_loss)
            # 计算focal loss
            focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
            if self.reduction == 'sum':
                return focal.sum()
            # 缺省：按有效token数做mean
            return focal.mean()
        else:
            # 没有有效位置，返回0保持可导
            return inputs.new_zeros([], dtype=inputs.dtype).sum()

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class WarmupCosineScheduler:
    """Warmup + Cosine退火调度器"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine退火阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

# 创建output目录
os.makedirs('output', exist_ok=True)

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'output/training_{timestamp}.log'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _load_checkpoint_robust(path: str):
    """在不同PyTorch版本下稳健加载checkpoint。
    优先用默认torch.load；失败则尝试允许旧pickle所需的safe globals并设置weights_only=False。
    返回(ckpt, mode_str)。
    """
    # 尝试默认加载
    try:
        ckpt = torch.load(path, map_location='cpu')
        return ckpt, 'default'
    except Exception as e1:
        last_err = e1
    # 回退：允许numpy旧构造器，并禁用weights_only
    try:
        try:
            import torch.serialization as _ts
            import numpy as _np
            if hasattr(_ts, 'add_safe_globals'):
                try:
                    _ts.add_safe_globals([_np.core.multiarray._reconstruct])
                except Exception:
                    pass
        except Exception:
            pass
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        logger.info("使用 weights_only=False 加载checkpoint（兼容旧格式）")
        return ckpt, 'weights_only_false'
    except TypeError:
        # 老版本torch.load不支持weights_only参数，重试默认加载
        ckpt = torch.load(path, map_location='cpu')
        return ckpt, 'default_retry'
    except Exception as e2:
        # 双重失败，抛出供上层处理
        raise e2

def _get_main_process_flag() -> bool:
    try:
        return int(os.environ.get('RANK', '0')) == 0
    except Exception:
        return True

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: 'WarmupCosineScheduler', epoch: int, config: dict,
                    best_val_loss: float | None = None, early_stopping_state: dict | None = None,
                    extra: dict | None = None, current_step: int = 0):
    """仅主进程保存完整断点（模型+优化器+调度器+随机态）。"""
    is_main = _get_main_process_flag()
    if not is_main:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {}
    # 模型权重
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    to_save['model_state_dict'] = module.state_dict()
    # 优化器
    to_save['optimizer_state_dict'] = optimizer.state_dict()
    # 调度器完整状态
    to_save['scheduler_state'] = {
        'current_step': getattr(scheduler, 'current_step', current_step),
        'base_lr': getattr(scheduler, 'base_lr', optimizer.param_groups[0]['lr']),
        'warmup_steps': getattr(scheduler, 'warmup_steps', 0),
        'total_steps': getattr(scheduler, 'total_steps', 0),
        'min_lr': getattr(scheduler, 'min_lr', 0.0),
    }
    # 训练元信息
    to_save['epoch'] = epoch
    to_save['config'] = config
    to_save['best_val_loss'] = best_val_loss
    to_save['early_stopping'] = early_stopping_state or {}
    to_save['global_step'] = current_step  # 添加全局步数记录
    # 随机数状态（保证可重复与平滑续训）
    to_save['rng_state'] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # 其他扩展（如动态微批大小等）
    if extra:
        to_save.update(extra)
    torch.save(to_save, path)

def save_step_checkpoint(global_step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: 'WarmupCosineScheduler', epoch: int, config: dict,
                        best_val_loss: float | None = None, early_stopping_state: dict | None = None,
                        extra: dict | None = None, keep_last_n: int = 3):
    """
    按步数保存检查点，自动清理旧检查点
    Args:
        global_step: 全局步数
        keep_last_n: 保留最近N个检查点（默认3个）
    """
    is_main = _get_main_process_flag()
    if not is_main:
        return
    
    # 保存当前步数检查点
    step_checkpoint_path = f'output/checkpoint_step_{global_step}.pth'
    save_checkpoint(
        path=step_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        config=config,
        best_val_loss=best_val_loss,
        early_stopping_state=early_stopping_state,
        extra=extra,
        current_step=global_step
    )
    
    # 自动清理旧检查点
    try:
        output_dir = 'output'
        # 找到所有步数检查点文件
        checkpoint_files = []
        for f in os.listdir(output_dir):
            if f.startswith('checkpoint_step_') and f.endswith('.pth'):
                try:
                    step_num = int(f.replace('checkpoint_step_', '').replace('.pth', ''))
                    checkpoint_files.append((step_num, f))
                except ValueError:
                    continue
        
        # 按步数排序，保留最新的keep_last_n个
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        files_to_delete = checkpoint_files[keep_last_n:]
        
        for step_num, filename in files_to_delete:
            file_path = os.path.join(output_dir, filename)
            try:
                os.remove(file_path)
                logger.info(f"🗑️ 删除旧检查点: {filename}")
            except Exception as e:
                logger.warning(f"删除检查点失败 {filename}: {e}")
                
    except Exception as e:
        logger.warning(f"清理检查点时出错: {e}")

def cleanup_best_checkpoints(keep_k: int = 3, out_dir: str = 'output') -> None:
    """
    仅保留 val_loss 最低的前 keep_k 个 best_pretrain_model_step_*.pth，余者删除。
    """
    try:
        files = [f for f in os.listdir(out_dir) if f.startswith('best_pretrain_model_step_') and f.endswith('.pth')]
        if len(files) <= keep_k:
            return
        scored: list[tuple[float, str]] = []
        for fname in files:
            fpath = os.path.join(out_dir, fname)
            val_loss = float('inf')
            try:
                meta = torch.load(fpath, map_location='cpu')
                val_loss = float(meta.get('val_loss', val_loss))
            except Exception:
                pass
            scored.append((val_loss, fpath))
        scored.sort(key=lambda x: x[0])  # val_loss越小越好
        to_delete = scored[keep_k:]
        for _vl, path in to_delete:
            try:
                os.remove(path)
                logger.info(f"🧹 删除旧best: {os.path.basename(path)}")
            except Exception as e:
                logger.warning(f"删除旧best失败 {os.path.basename(path)}: {e}")
    except Exception as e:
        logger.warning(f"清理best时出错: {e}")

def cleanup_best_checkpoints_by_metric(prefix: str, meta_key: str, keep_k: int = 3, out_dir: str = 'output') -> None:
    """
    按任意度量(meta_key)仅保留前 keep_k 个 best_{metric} 权重。
    例如：prefix='best_pretrain_model_rmse_step_', meta_key='flux_rmse'
    """
    try:
        files = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith('.pth')]
        if len(files) <= keep_k:
            return
        scored: list[tuple[float, str]] = []
        for fname in files:
            fpath = os.path.join(out_dir, fname)
            score = float('inf')
            try:
                meta = torch.load(fpath, map_location='cpu')
                # 越小越好
                score = float(meta.get(meta_key, score))
            except Exception:
                pass
            scored.append((score, fpath))
        scored.sort(key=lambda x: x[0])
        to_delete = scored[keep_k:]
        for _val, path in to_delete:
            try:
                os.remove(path)
                logger.info(f"🧹 删除旧best({meta_key}): {os.path.basename(path)}")
            except Exception as e:
                logger.warning(f"删除旧best({meta_key})失败 {os.path.basename(path)}: {e}")
    except Exception as e:
        logger.warning(f"清理best({meta_key})时出错: {e}")

def _find_latest_checkpoint() -> str | None:
    out_dir = 'output'
    last_ckpt = os.path.join(out_dir, 'last_checkpoint.pth')
    if os.path.isfile(last_ckpt):
        return last_ckpt
    # 回退：寻找最高epoch的checkpoint-epoch*.pth（仍保留以向后兼容，但默认不再写入该类文件）
    try:
        files = [f for f in os.listdir(out_dir) if f.startswith('checkpoint-epoch') and f.endswith('.pth')]
        if not files:
            return None
        def _epoch_num(name: str) -> int:
            try:
                s = name.replace('checkpoint-epoch', '').replace('.pth', '')
                return int(s)
            except Exception:
                return -1
        files.sort(key=_epoch_num, reverse=True)
        return os.path.join(out_dir, files[0])
    except Exception:
        return None

def _stable_mod_by_world(value, world_size: int) -> int:
    """对spectrum_id做稳定哈希，避免Python进程间随机种子影响。"""
    bs = str(value).encode('utf-8')
    h = hashlib.md5(bs).hexdigest()
    return int(h[:8], 16) % max(1, world_size)

def _build_shards_by_spectrum_id(input_csv: str, shard_dir: str, world_size: int, is_main_process: bool, label: str):
    """Rank0使用流式分片，将input_csv按spectrum_id稳定哈希切成world_size个CSV，其他rank等待。"""
    if world_size <= 1:
        return
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, f"{label}_rank{i}.csv") for i in range(world_size)]
    done_flag = os.path.join(shard_dir, f"{label}.done")
    if is_main_process:
        for p in shard_paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        # 清理完成标志
        try:
            if os.path.isfile(done_flag):
                os.remove(done_flag)
        except Exception:
            pass
        logger.info(f"开始创建磁盘分片: {input_csv} -> {shard_dir} ({world_size} 份)")
        # 获取需要的列（若不存在则自动忽略）
        try:
            header_df = pd.read_csv(input_csv, nrows=0)
            all_cols = set(header_df.columns.tolist())
        except Exception as e:
            logger.warning(f"读取列名失败: {e}")
            all_cols = set()
        needed = {'spectrum_id','pixel_idx','flux_thu','flux_hun','flux_ten','flux_one','teff','logg','feh'}
        usecols = list(needed & all_cols) if all_cols else None
        chunk_size = int(os.environ.get('CSV_CHUNK_SIZE', '1000000'))
        total_rows = 0
        csv_engine = os.environ.get('CSV_ENGINE', '').strip()
        read_kwargs = {
            'chunksize': chunk_size,
        }
        if usecols:
            read_kwargs['usecols'] = usecols
        if csv_engine:
            read_kwargs['engine'] = csv_engine
        for chunk in pd.read_csv(input_csv, **read_kwargs):
            total_rows += len(chunk)
            idxs = chunk['spectrum_id'].map(lambda v: _stable_mod_by_world(v, world_size))
            for i in range(world_size):
                sub = chunk[idxs == i]
                if len(sub) == 0:
                    continue
                out_path = shard_paths[i]
                write_header = not os.path.exists(out_path)
                sub.to_csv(out_path, mode='a', header=write_header, index=False)
        logger.info(f"磁盘分片完成，共处理 {total_rows} 行；输出: {len(shard_paths)} 文件")
        # 写完成标志
        try:
            with open(done_flag, 'w') as f:
                f.write('done')
        except Exception:
            pass
    # 文件系统轮询等待，避免NCCL barrier在长时间IO时超时
    start_wait = time.time()
    timeout_sec = int(os.environ.get('SHARD_WAIT_TIMEOUT_SEC', '7200'))
    while True:
        ready = os.path.isfile(done_flag)
        # 可选：也检查本rank分片存在且非空（若调用方需要）
        if ready:
            break
        if (time.time() - start_wait) > timeout_sec:
            raise TimeoutError(f"等待{label}分片完成超时（>{timeout_sec}s）。请检查共享存储与rank0状态")
        time.sleep(2)

class StreamlinedPreprocessor:
    """简化但高效的预处理器"""
    
    def __init__(self, token_to_id, id_to_token, seq_len=8192):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.seq_len = seq_len
        self.bos_token = token_to_id.get('<BOS>', 0)
        self.eos_token = token_to_id.get('<EOS>', 1)
        self.pad_token = token_to_id.get('<SEP>', 2)
        # none占位：优先[None]，否则退回<SEP>，避免行为变化
        self.none_token = token_to_id.get('[None]', self.pad_token)
    
    def preprocess_fast(self, df, max_samples=3100, is_main_process: bool = True):
        """快速预处理"""
        if is_main_process:
            logger.info(f"🚀 开始快速预处理 {len(df)} 行数据")
        
        # 按spectrum_id分组
        grouped = df.groupby('spectrum_id')
        if is_main_process:
            logger.info(f"光谱数量: {len(grouped)}")
        
        results = []
        flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']
        
        # 使用tqdm显示进度
        spectrum_groups = list(grouped)[:max_samples]
        
        # 旧阶段可覆盖的最大像素：默认按 seq_len 推导 (floor((seq_len-2)/4)-1)
        try:
            _pmax_env = os.environ.get('POS_MAX_PIXEL', '').strip()
            if _pmax_env:
                pos_max_pixel = int(_pmax_env)
            else:
                pos_max_pixel = max(0, (self.seq_len - 2) // 4 - 1)
        except Exception:
            pos_max_pixel = max(0, (self.seq_len - 2) // 4 - 1)

        for spectrum_id, group in tqdm(spectrum_groups, desc="预处理光谱", disable=(not is_main_process)):
            try:
                # 按pixel_idx排序
                group = group.sort_values('pixel_idx')
                
                # 向量化提取flux tokens
                flux_matrix = group[flux_columns].values
                flux_tokens = []
                # 绝对位置索引构造（digit级）：像素索引*4 + digit_offset(0..3)
                pos_index_core: list[int] = []
                
                for pix_idx, row in enumerate(flux_matrix):
                    # pixel_idx 来自 group 排序后的列
                    try:
                        p = int(group.iloc[pix_idx]['pixel_idx'])
                    except Exception:
                        p = 0
                    # 过滤旧阶段未覆盖的像素
                    if p > pos_max_pixel:
                        continue
                    for j, token in enumerate(row):
                        if pd.notna(token) and token in self.token_to_id:
                            flux_tokens.append(self.token_to_id[token])
                        else:
                            flux_tokens.append(self.none_token)
                        # 与流式数据一致：每像素的4个digit共享同一像素级绝对索引
                        pos_index_core.append(p)
                
                # 构建序列
                sequence = [self.bos_token] + flux_tokens + [self.eos_token]
                
                # 截断或填充
                if len(sequence) > self.seq_len:
                    sequence = sequence[:self.seq_len]
                else:
                    sequence.extend([self.pad_token] * (self.seq_len - len(sequence)))

                # 位置索引：直接使用像素原始索引p（每像素4个digit共用同一p），对齐到seq_len
                block_size = self.seq_len
                if pos_index_core:
                    pos_index_arr = np.array(pos_index_core, dtype=np.int64)
                    pos_index_arr = np.clip(pos_index_arr, 0, block_size - 1)
                    pos_index_seq = [0] + pos_index_arr[: max(0, self.seq_len - 2)].tolist() + [0]
                    if len(pos_index_seq) < self.seq_len:
                        pos_index_seq.extend([0] * (self.seq_len - len(pos_index_seq)))
                else:
                    pos_index_seq = [0] * self.seq_len
                
                # 提取参数
                params = None
                if 'teff' in group.columns:
                    first_row = group.iloc[0]
                    params = [
                        float(first_row.get('teff', 5000.0)),
                        float(first_row.get('logg', 4.5)),
                        float(first_row.get('feh', 0.0))
                    ]
                
                results.append({
                    'spectrum_id': spectrum_id,
                    'sequence': sequence,
                    'pos_index': pos_index_seq,
                    'params': params,
                    'flux_count': len(flux_tokens) // 4
                })
                
            except Exception as e:
                logger.warning(f"处理光谱 {spectrum_id} 时出错: {e}")
                continue
        
        if is_main_process:
            logger.info(f"✅ 预处理完成: {len(results)} 个有效样本")
        return results

class StreamlinedDataset(Dataset):
    """简化数据集（初始化阶段缓存Tensor，__getitem__零分配）"""
    
    def __init__(self, data):
        # 预先转换并缓存到CPU内存（按批pin由DataLoader完成）
        sequences: list[torch.Tensor] = []
        params_list: list[torch.Tensor | None] = []
        pos_index_list: list[torch.Tensor] = []
        spectrum_ids: list = []
        flux_counts: list[int] = []

        for item in data:
            sequences.append(torch.as_tensor(item['sequence'], dtype=torch.long))
            spectrum_ids.append(item['spectrum_id'])
            flux_counts.append(item['flux_count'])
            # pos_index 同shape，dtype long
            if 'pos_index' in item:
                pos_index_list.append(torch.as_tensor(item['pos_index'], dtype=torch.long))
            else:
                pos_index_list.append(torch.zeros_like(sequences[-1], dtype=torch.long))
            if item.get('params') is not None:
                params_list.append(torch.as_tensor(item['params'], dtype=torch.float))
            else:
                params_list.append(None)

        # 释放原始data引用，降低内存占用
        del data

        self.sequences = sequences
        self.params_list = params_list
        self.pos_index_list = pos_index_list
        self.spectrum_ids = spectrum_ids
        self.flux_counts = flux_counts
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        result = {
            'sequence': self.sequences[idx],
            'spectrum_id': self.spectrum_ids[idx],
            'flux_count': self.flux_counts[idx],
            'pos_index': self.pos_index_list[idx],
        }
        param_tensor = self.params_list[idx]
        if param_tensor is not None:
            result['params'] = param_tensor
        return result

class StreamedSpectrumIterableDataset(torch.utils.data.IterableDataset):
    """基于CSV的流式数据集：逐个spectrum_id产出样本，避免整表载入内存。
    重要假设：CSV中同一spectrum_id的行在文件中是连续的（常见导出格式）。
    若跨chunk边界，内部通过carry-over将同一id在相邻chunk拼接后再产出。
    支持：
      - 按rank在线筛选（稳定哈希到world_size，避免磁盘分片）
      - 按稳定哈希进行9:1等比例划分，保证同一spectrum_id不会跨split
    """

    def __init__(self, csv_path: str, preprocessor: StreamlinedPreprocessor,
                 max_samples: int | None, is_main_process: bool,
                 filter_by_rank: bool = False, world_size: int = 1, global_rank: int = 0,
                 split_by_hash: bool = False, split_mod_base: int = 10, split_threshold: int = 9,
                 is_train_split: bool = True):
        super().__init__()
        self.csv_path = csv_path
        self.preprocessor = preprocessor
        self.max_samples = max_samples
        self.is_main_process = is_main_process
        # 在线分片/筛选配置
        self.filter_by_rank = bool(filter_by_rank)
        self.world_size = int(world_size)
        self.global_rank = int(global_rank)
        # 稳定哈希划分配置
        self.split_by_hash = bool(split_by_hash)
        self.split_mod_base = int(split_mod_base)
        self.split_threshold = int(split_threshold)
        self.is_train_split = bool(is_train_split)

    def __iter__(self):
        token_to_id = self.preprocessor.token_to_id
        id_to_token = self.preprocessor.id_to_token
        seq_len = self.preprocessor.seq_len
        bos = self.preprocessor.bos_token
        eos = self.preprocessor.eos_token
        pad = self.preprocessor.pad_token
        none_tok = self.preprocessor.none_token
        flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']

        # 读取块大小与列	s
        env_chunk = os.environ.get('CSV_CHUNK_SIZE', '').strip()
        if env_chunk:
            try:
                chunk_size = int(env_chunk)
            except Exception:
                chunk_size = 200000
        else:
            # 与分片函数相同的H800默认逻辑
            h800_detected = False
            try:
                if torch.cuda.is_available():
                    name0 = torch.cuda.get_device_name(0)
                    if 'H800' in str(name0).upper():
                        h800_detected = True
            except Exception:
                pass
            chunk_size = 400000 if h800_detected else 200000

        produced = 0
        current_id = None
        current_rows = []
        current_include = False

        # DataLoader多worker支持：使用稳定哈希将不同spectrum_id分配到不同worker，避免同一id跨worker拆分
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # 推迟导入pandas读取块
        csv_engine = os.environ.get('CSV_ENGINE', '').strip()
        read_kwargs = {
            'chunksize': chunk_size,
        }
        # PyArrow 引擎不支持 chunksize，回退到默认引擎
        if csv_engine and csv_engine.lower() != 'pyarrow':
            read_kwargs['engine'] = csv_engine
        reader = pd.read_csv(self.csv_path, **read_kwargs)
        for chunk in reader:
            # 仅保留需要列（若缺失自动忽略）
            have_cols = set(chunk.columns.tolist())
            need_cols = { 'spectrum_id', 'pixel_idx', *flux_columns, 'teff', 'logg', 'feh' }
            keep_cols = list(have_cols & need_cols)
            if len(keep_cols) != len(chunk.columns):
                chunk = chunk[keep_cols]
            # 确保按spectrum_id、pixel_idx顺序（若已排序无影响）
            assume_sorted = os.environ.get('ASSUME_SORTED', '0') == '1'
            if not assume_sorted:
                if 'pixel_idx' in chunk.columns:
                    chunk = chunk.sort_values(['spectrum_id','pixel_idx'])
                else:
                    chunk = chunk.sort_values(['spectrum_id'])

            for _, row in chunk.iterrows():
                sid = row['spectrum_id']
                if current_id is None:
                    current_id = sid
                    # 统一首个样本的worker过滤参数，避免首批不一致
                    current_include = self._should_include_sid(sid, num_workers=num_workers, worker_id=worker_id)
                if sid != current_id:
                    # 产出上一id（仅当通过筛选）
                    if current_include and current_rows:
                        yield from self._emit_sample(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)
                        produced += 1
                        if self.max_samples is not None and produced >= self.max_samples:
                            return
                    current_id = sid
                    current_rows = []
                    current_include = self._should_include_sid(sid, num_workers=num_workers, worker_id=worker_id)
                # 行级加入worker筛选后的样本
                if current_include:
                    current_rows.append(row)

        # 文件结束，产出最后一个id
        if current_id is not None and current_rows and current_include:
            yield from self._emit_sample(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)

    def _should_include_sid(self, spectrum_id_value, num_workers: int = 1, worker_id: int = 0) -> bool:
        """基于稳定哈希与rank筛选，决定是否保留该spectrum_id。"""
        # 先按split划分，确保同一id不会跨split
        if self.split_by_hash:
            # 使用与_rank分片相同风格的稳定哈希（md5前8位），但可配置mod基数
            bs = str(spectrum_id_value).encode('utf-8')
            h = hashlib.md5(bs).hexdigest()
            modv = int(h[:8], 16) % max(1, self.split_mod_base)
            in_train = (modv < self.split_threshold)
            if self.is_train_split and not in_train:
                return False
            if (not self.is_train_split) and in_train:
                return False
        # 再做rank筛选，避免重复数据
        if self.filter_by_rank and self.world_size > 1:
            if _stable_mod_by_world(spectrum_id_value, self.world_size) != (self.global_rank % self.world_size):
                return False
        # 最后做worker内部分配，避免同一id跨worker拆分（每个worker仍各自读取文件，但只保留自己负责的id）
        if num_workers > 1:
            if _stable_mod_by_world(spectrum_id_value, num_workers) != (worker_id % num_workers):
                return False
        return True

    def _emit_sample(self, spectrum_id, rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len):
        
        try:
            df = pd.DataFrame(rows)
            if 'pixel_idx' in df.columns:
                df = df.sort_values('pixel_idx')
            flux_tokens = []
            # 过滤阈值：与训练预处理一致
            try:
                _pmax_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                if _pmax_env:
                    pos_max_pixel = int(_pmax_env)
                else:
                    pos_max_pixel = max(0, (seq_len - 2) // 4 - 1)
            except Exception:
                pos_max_pixel = max(0, (seq_len - 2) // 4 - 1)
            # 按像素逐行构造，仅保留 <=阈值 的像素
            if 'pixel_idx' in df.columns and all(col in df.columns for col in flux_columns):
                for _, r in df.iterrows():
                    try:
                        p = int(r['pixel_idx'])
                    except Exception:
                        p = 0
                    if p > pos_max_pixel:
                        continue
                    for col in flux_columns:
                        token = r[col]
                        if pd.notna(token) and token in token_to_id:
                            flux_tokens.append(token_to_id[token])
                        else:
                            flux_tokens.append(none_tok)
            else:
                flux_matrix = df[flux_columns].values if all(col in df.columns for col in flux_columns) else np.empty((0,4), dtype=object)
                for row in flux_matrix:
                    for token in row:
                        if pd.notna(token) and token in token_to_id:
                            flux_tokens.append(token_to_id[token])
                        else:
                            flux_tokens.append(none_tok)
            sequence = [bos] + flux_tokens + [eos]
            if len(sequence) > seq_len:
                sequence = sequence[:seq_len]
            else:
                sequence.extend([pad] * (seq_len - len(sequence)))
            params = None
            if 'teff' in df.columns:
                first = df.iloc[0]
                params = [float(first.get('teff', 5000.0)), float(first.get('logg', 4.5)), float(first.get('feh', 0.0))]
            # 绝对位置索引（pos_index）：直接使用像素编号p（digit级重复），BOS/EOS/PAD位置填0
            block_size = int(os.environ.get('BLOCK_SIZE', str(self.preprocessor.seq_len)))
            if 'pixel_idx' in df.columns and len(flux_tokens) > 0:
                pix = df['pixel_idx'].astype(int).tolist()
                # 过滤旧阶段未覆盖的像素
                try:
                    _pmax_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                    if _pmax_env:
                        pos_max_pixel = int(_pmax_env)
                    else:
                        pos_max_pixel = max(0, (self.preprocessor.seq_len - 2) // 4 - 1)
                except Exception:
                    pos_max_pixel = max(0, (self.preprocessor.seq_len - 2) // 4 - 1)
                pix = [p for p in pix if p <= pos_max_pixel]
                # 1) 像素级 → digit级绝对坐标（flatten）
                abs_token = []
                for p in pix:
                    abs_token.extend([p] * 4)
                # 2) 直接裁剪到 0..(block_size-1)
                pos_index = np.array(abs_token, dtype=np.int64)
                pos_index = np.clip(pos_index, 0, block_size - 1)
                # 补 BOS/EOS，并对齐到 seq_len
                pos_index = [0] + pos_index[: max(0, self.preprocessor.seq_len - 2)].tolist() + [0]
                if len(pos_index) < self.preprocessor.seq_len:
                    pos_index.extend([0] * (self.preprocessor.seq_len - len(pos_index)))
            else:
                pos_index = [0] * self.preprocessor.seq_len

            sample = {
                'sequence': torch.as_tensor(sequence, dtype=torch.long),
                'spectrum_id': spectrum_id,
                'flux_count': len(flux_tokens) // 4,
                'pos_index': torch.as_tensor(pos_index, dtype=torch.long),
            }
            if params is not None:
                sample['params'] = torch.as_tensor(params, dtype=torch.float)
            yield sample
        except Exception:
            return

class FluxRMSECalculator:
    """简化的Flux RMSE计算器"""
    
    def __init__(self, token_to_id, id_to_token):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        # 控制采样规模，降低方差且避免过慢
        try:
            import os as _os
            self.max_groups = int(_os.environ.get('RMSE_MAX_GROUPS', '256'))  # 每样本最多取多少个4位组
            self.max_batch_samples = int(_os.environ.get('RMSE_BATCH_SAMPLES', '8'))  # 每批最多取多少样本
        except Exception:
            self.max_groups = 256
            self.max_batch_samples = 8
        
    def calculate(self, pred_logits, targets):
        """计算flux RMSE"""
        try:
            if pred_logits.shape[1] > targets.shape[1]:
                pred_logits = pred_logits[:, :-1, :]
            
            pred_tokens = torch.argmax(pred_logits, dim=-1)
            
            pred_flux = []
            true_flux = []
            
            batch_size = targets.shape[0]
            # 扩大样本数但保持上限，降低方差
            for b in range(min(batch_size, self.max_batch_samples)):
                pred_seq = pred_tokens[b].cpu().numpy()
                true_seq = targets[b].cpu().numpy()
                
                # 从真实序列中定位首个S*起点，按4步对齐抽组，遇到非S(<EOS>/<SEP>)即停止
                # 若未找到S*，跳过该样本
                start_idx = None
                for pos in range(len(true_seq)):
                    tok = self.id_to_token.get(int(true_seq[pos]), None)
                    if isinstance(tok, str) and tok.startswith('S'):
                        start_idx = pos
                        break
                if start_idx is None:
                    continue

                groups_taken = 0
                max_i = min(len(pred_seq), start_idx + 4 * self.max_groups)
                for i in range(start_idx, max_i, 4):
                    if i + 3 >= len(pred_seq):
                        break
                    try:
                        pred_digits = []
                        true_digits = []
                        valid_group = True
                        for j in range(4):
                            ptid = int(pred_seq[i + j])
                            ptok = self.id_to_token.get(ptid, None)
                            ttid = int(true_seq[i + j])
                            ttok = self.id_to_token.get(ttid, None)
                            # 只接受S*，否则认为序列结束
                            if not (isinstance(ptok, str) and ptok.startswith('S')):
                                valid_group = False
                            if not (isinstance(ttok, str) and ttok.startswith('S')):
                                valid_group = False
                            if valid_group:
                                pred_digits.append(ptok[1])
                                true_digits.append(ttok[1])
                        if not valid_group:
                            break
                        if len(pred_digits) == 4 and len(true_digits) == 4:
                            pred_val = int(''.join(pred_digits))
                            true_val = int(''.join(true_digits))
                            pred_flux.append(pred_val)
                            true_flux.append(true_val)
                            groups_taken += 1
                    except Exception:
                        break
            
            if len(pred_flux) > 0:
                pred_flux = np.array(pred_flux)
                true_flux = np.array(true_flux)
                rmse = np.sqrt(np.mean((pred_flux - true_flux) ** 2))
                return rmse
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"计算Flux RMSE时出错: {e}")
            return 0.0

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, enable_plots: bool = False):
        self.history = {
            'pretrain': {
                'epochs': [], 'train_loss': [], 'val_loss': [],
                'flux_rmse': [], 'val_flux_rmse': []
            },
            'finetune': {
                'epochs': [], 'train_loss': [], 'val_loss': [],
                'param_rmse': [], 'val_param_rmse': []
            }
        }
        self.start_time = time.time()
        self.enable_plots = enable_plots
    
    def log_pretrain(self, epoch, train_loss, val_loss, flux_rmse, val_flux_rmse):
        """记录预训练"""
        self.history['pretrain']['epochs'].append(epoch)
        self.history['pretrain']['train_loss'].append(train_loss)
        self.history['pretrain']['val_loss'].append(val_loss)
        self.history['pretrain']['flux_rmse'].append(flux_rmse)
        self.history['pretrain']['val_flux_rmse'].append(val_flux_rmse)
        
        logger.info(f"预训练 Epoch {epoch}:")
        logger.info(f"  训练损失: {train_loss:.4f}, Flux RMSE: {flux_rmse:.2f}")
        logger.info(f"  验证损失: {val_loss:.4f}, Flux RMSE: {val_flux_rmse:.2f}")
        
        # 去除绘图
    
    def log_finetune(self, epoch, train_loss, val_loss, param_rmse, val_param_rmse):
        """记录微调"""
        self.history['finetune']['epochs'].append(epoch)
        self.history['finetune']['train_loss'].append(train_loss)
        self.history['finetune']['val_loss'].append(val_loss)
        self.history['finetune']['param_rmse'].append(param_rmse)
        self.history['finetune']['val_param_rmse'].append(val_param_rmse)
        
        logger.info(f"微调 Epoch {epoch}:")
        logger.info(f"  训练损失: {train_loss:.4f}, 参数RMSE: {param_rmse['total']:.4f}")
        logger.info(f"  验证损失: {val_loss:.4f}, 参数RMSE: {val_param_rmse['total']:.4f}")
        
        # 去除绘图
    
    def plot_pretrain(self):
        return
    
    def plot_finetune(self):
        return

def load_token_mapping():
    """加载token映射"""
    # 优先从环境变量读取（由启动脚本提供）
    vocab_path = os.environ.get('VOCAB_PATH', '').strip()
    if not vocab_path:
        raise FileNotFoundError("未设置 VOCAB_PATH 环境变量，请在启动脚本中导出 VOCAB_PATH=绝对路径")
    vocab_df = pd.read_csv(vocab_path)
    # 兼容无 token_id 列的词表：按出现顺序自动编号
    if 'token_id' in vocab_df.columns:
        token_to_id = dict(zip(vocab_df['token'], vocab_df['token_id']))
    else:
        tokens = vocab_df['token'].tolist()
        token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return token_to_id, id_to_token

def pretrain_phase(model, train_loader, val_loader, val_loader_full, config, flux_calc, monitor):
    """预训练阶段 - 集成高级训练技巧"""
    logger.info("🚀 开始预训练阶段 - 专注于flux生成（集成Focal Loss + 早停 + Warmup）")
    
    device = next(model.parameters()).device
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = (rank == 0)
    
    # 优化器（保持原逻辑，不启用fused以避免潜在数值差异）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['pretrain_lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # 总优化步数（保持每batch一步）。IterableDataset 无法取 len()，使用估计步数兜底
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = int(os.environ.get('ESTIMATED_STEPS_PER_EPOCH', '1000'))
        if is_main_process:
            logger.warning(
                f"len(train_loader) 不可用(IterableDataset)。使用估计 steps_per_epoch={steps_per_epoch}，"
                f"可通过环境变量 ESTIMATED_STEPS_PER_EPOCH 覆盖"
            )
    total_steps = max(1, config['pretrain_epochs'] * steps_per_epoch)
    warmup_ratio = float(config.get('warmup_ratio', 0.1))
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    
    # Warmup + Cosine调度器
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=(config['pretrain_lr'] * float(config.get('min_lr_ratio', 0.01)))
    )
    
    # Focal Loss（支持环境/配置覆盖超参）
    try:
        focal_alpha = float(os.environ.get('FOCAL_ALPHA', str(config.get('focal_alpha', 1.0))))
    except Exception:
        focal_alpha = float(config.get('focal_alpha', 1.0))
    try:
        focal_gamma = float(os.environ.get('FOCAL_GAMMA', str(config.get('focal_gamma', 2.0))))
    except Exception:
        focal_gamma = float(config.get('focal_gamma', 2.0))
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

    # AMP配置（默认开启bf16，仅影响数值精度路径，不改变步数）
    use_amp = (device.type == 'cuda') and bool(config.get('use_amp', True))
    amp_dtype = torch.bfloat16 if str(config.get('amp_dtype', 'bf16')).lower() == 'bf16' else torch.float16
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.get('patience', 5),
        min_delta=float(os.environ.get('ES_MIN_DELTA', '0.001')),
        restore_best_weights=True
    )
    
    # 提供一个简洁的token准确率统计（忽略 BOS/EOS/PAD，并限制评估token数以加速）
    def _token_accuracy_counts(logits: torch.Tensor, targets: torch.Tensor,
                               pad_id: int, bos_id: int, eos_id: int,
                               max_positions: int | None = 4096) -> tuple[int, int]:
        with torch.no_grad():
            # logits: [B, T, V], targets: [B, T]
            if max_positions is not None and logits.size(1) > max_positions:
                logits = logits[:, :max_positions, :]
                targets = targets[:, :max_positions]
            pred_tokens = torch.argmax(logits, dim=-1)
            mask = (targets != pad_id) & (targets != bos_id) & (targets != eos_id)
            correct = (pred_tokens == targets) & mask
            return int(correct.sum().item()), int(mask.sum().item())

    best_val_loss = float('inf')  # 仅用于日志和早停，不再用于best
    best_val_rmse = float('inf')
    last_epoch_val_loss = None  # 记录最后一个epoch的验证损失
    global_step = 0  # 全局步数计数器
    # 步数级检查点（2000步间隔，优先级最高；按需通过环境变量覆盖）
    step_checkpoint_interval = int(os.environ.get('STEP_CHECKPOINT_INTERVAL', '2000'))
    enable_step_ckpt = (os.environ.get('ENABLE_STEP_CHECKPOINTS', '1') == '1')
    keep_n_checkpoints = int(os.environ.get('KEEP_N_CHECKPOINTS', '6'))
    keep_k_best_rmse = int(os.environ.get('KEEP_K_BEST_RMSE', os.environ.get('KEEP_K_BEST', '3')))
    # 仅保留步数级与last/最优；移除冗余的epoch级保留数量设置

    # 断点续训：可通过 RESUME_FROM=path 或 AUTO_RESUME=1 启动
    resume_path = os.environ.get('RESUME_FROM', '').strip()
    if not resume_path and os.environ.get('AUTO_RESUME', '0') == '1':
        resume_path = _find_latest_checkpoint() or ''
    start_epoch = 0
    resume_misc = {}
    # 实际加载断点并恢复模型/全局步数（支持best/step/last）
    if resume_path:
        try:
            if is_main_process:
                logger.info(f"🔄 从断点恢复: {resume_path}")
            ckpt, _load_mode = _load_checkpoint_robust(resume_path)
            if isinstance(ckpt, dict):
                resume_misc = ckpt
                state = ckpt.get('model_state_dict', None)
                if state is not None:
                    target_module = (model.module if isinstance(model, DDP) else model)
                    strict_resume = os.environ.get('RESUME_STRICT', '1') == '1'
                    missing, unexpected = target_module.load_state_dict(state, strict=strict_resume)
                    if is_main_process:
                        if not strict_resume and (missing or unexpected):
                            logger.warning(f"部分权重未加载(missing={len(missing)}, unexpected={len(unexpected)})")
                # 恢复全局步数（用于步级验证/保存命名的延续）
                try:
                    global_step = int(ckpt.get('global_step', 0))
                except Exception:
                    global_step = 0
                # 兼容旧版checkpoint：若缺失global_step，则从文件名或调度器恢复
                if (not isinstance(global_step, int)) or global_step <= 0:
                    try:
                        import re as _re
                        m = _re.search(r"checkpoint_step_(\\d+)\\.pth$", str(resume_path))
                        if m:
                            global_step = int(m.group(1))
                            if is_main_process:
                                logger.info(f"从文件名推断global_step={global_step}")
                        elif 'scheduler_state' in ckpt and isinstance(ckpt['scheduler_state'], dict):
                            gs_fallback = int(ckpt['scheduler_state'].get('current_step', 0))
                            if gs_fallback > 0:
                                global_step = gs_fallback
                                if is_main_process:
                                    logger.info(f"从scheduler_state恢复global_step={global_step}")
                    except Exception:
                        pass
                if is_main_process:
                    logger.info(f"恢复全局步数: global_step={global_step}")
        except Exception as e:
            if is_main_process:
                logger.warning(f"加载断点失败，将从头训练: {e}")
    
    logger.info(f"📋 训练配置:")
    logger.info(f"   总步数: {total_steps}")
    logger.info(f"   Warmup步数: {warmup_steps}")
    logger.info(f"   早停耐心: {config.get('patience', 5)}")
    logger.info(f"   Focal Loss (α={focal_loss.alpha}, γ={focal_loss.gamma})")
    # 允许用环境变量覆盖训练中打印指标的频率（按“批次”计）
    metric_interval = int(os.environ.get('TRAIN_METRIC_INTERVAL', str(config.get('train_metric_interval', 50))))
    enable_step_compare = bool(config.get('enable_step_compare', False))

    # Random_CL 概率调度（默认关闭，保持向后兼容）
    try:
        cl_ratio_start = float(os.environ.get('CL_RATIO_START', str(config.get('cl_ratio_start', 0.8))))
    except Exception:
        cl_ratio_start = float(config.get('cl_ratio_start', 0.8))
    try:
        cl_ratio_end = float(os.environ.get('CL_RATIO_END', str(config.get('cl_ratio_end', cl_ratio_start))))
    except Exception:
        cl_ratio_end = float(config.get('cl_ratio_end', cl_ratio_start))
    cl_ratio_schedule = (os.environ.get('CL_RATIO_SCHEDULE', '0') == '1')
    cl_ratio_reset_on_resume = (os.environ.get('CL_RATIO_RESET_ON_RESUME', '1') == '1')

    def _current_cl_ratio() -> float:
        if not cl_ratio_schedule:
            return cl_ratio_start
        step_for_progress = scheduler.current_step if cl_ratio_reset_on_resume else global_step
        progress = min(1.0, max(0.0, step_for_progress / max(1, total_steps)))
        return cl_ratio_start + (cl_ratio_end - cl_ratio_start) * progress
    
    # 动态微批：保持一次优化器步对应DataLoader的一个batch，拆分到显存可承受的微批上顺序执行
    dynamic_micro_bs = None  # 首次根据OOM自动回退确定

    # 优化器、调度器在此函数内创建，因此在此处执行加载
    if resume_misc:
        ckpt = resume_misc
        # 恢复优化器（可通过 RESUME_LOAD_OPTIMIZER=0 跳过以节省显存）
        try:
            load_optim = (os.environ.get('RESUME_LOAD_OPTIMIZER', '1') == '1')
            if load_optim and 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if is_main_process:
                    logger.info("优化器状态已恢复")
            elif not load_optim:
                if is_main_process:
                    logger.info("跳过优化器状态恢复 (RESUME_LOAD_OPTIMIZER=0)")
        except Exception as e:
            if is_main_process:
                logger.warning(f"加载优化器状态失败: {e}")
        # 恢复调度器
        try:
            ss = ckpt.get('scheduler_state', {})
            scheduler.current_step = int(ss.get('current_step', scheduler.current_step))
            scheduler.base_lr = float(ss.get('base_lr', scheduler.base_lr))
            scheduler.warmup_steps = int(ss.get('warmup_steps', scheduler.warmup_steps))
            scheduler.total_steps = int(ss.get('total_steps', scheduler.total_steps))
            scheduler.min_lr = float(ss.get('min_lr', scheduler.min_lr))
            # 按当前step重设学习率而不推进步数
            if scheduler.current_step <= scheduler.warmup_steps:
                lr_now = scheduler.base_lr * max(1, scheduler.current_step) / max(1, scheduler.warmup_steps)
            else:
                progress = (scheduler.current_step - scheduler.warmup_steps) / max(1, (scheduler.total_steps - scheduler.warmup_steps))
                lr_now = scheduler.min_lr + (scheduler.base_lr - scheduler.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now
            if is_main_process:
                logger.info(f"调度器状态已恢复：current_step={scheduler.current_step}, lr={lr_now:.2e}")
        except Exception as e:
            if is_main_process:
                logger.warning(f"加载调度器状态失败: {e}")
        # start_epoch等已在前一个恢复块设置

        # 续训可选：重置学习率+重新warmup（适配低信噪比域迁移）
        try:
            reset_lr_on_resume = (os.environ.get('RESUME_RESET_LR', '0') == '1') or bool(config.get('resume_reset_lr', False))
            if reset_lr_on_resume:
                base_lr_src = os.environ.get('RESUME_LR', os.environ.get('PRETRAIN_LR', str(config.get('pretrain_lr', 2.6e-4))))
                new_base_lr = float(base_lr_src)
                lr_multiplier = float(os.environ.get('RESUME_LR_MULTIPLIER', '1.0'))
                new_base_lr = max(1e-8, new_base_lr * lr_multiplier)
                for pg in optimizer.param_groups:
                    pg['lr'] = new_base_lr
                if os.environ.get('CLEAR_OPTIM_MOMENTUM_ON_RESUME', '1') == '1':
                    try:
                        for st in optimizer.state.values():
                            if 'exp_avg' in st:
                                st['exp_avg'].zero_()
                            if 'exp_avg_sq' in st:
                                st['exp_avg_sq'].zero_()
                    except Exception:
                        pass
                try:
                    scheduler.base_lr = new_base_lr
                    resume_warm_ratio = float(os.environ.get('RESUME_WARMUP_RATIO', os.environ.get('WARMUP_RATIO', str(config.get('warmup_ratio', 0.1)))))
                    scheduler.warmup_steps = max(1, int(resume_warm_ratio * total_steps))
                    scheduler.current_step = 0
                    if is_main_process:
                        logger.info(f"续训重置学习率：base_lr={new_base_lr:.2e}, warmup_steps={scheduler.warmup_steps}")
                except Exception as e:
                    if is_main_process:
                        logger.warning(f"续训重置学习率失败: {e}")
        except Exception:
            pass

    # 准备特殊token id（用于准确率统计时屏蔽BOS/EOS/PAD）
    pad_token_id = int(config.get('pad_token_id', 2))
    bos_token_id = int(config.get('bos_token_id', 0))
    eos_token_id = int(config.get('eos_token_id', 1))
    # 训练损失忽略特殊token（保持总体设计不变，仅加屏蔽）
    try:
        focal_loss.set_ignore_token_ids({pad_token_id, bos_token_id, eos_token_id})
    except Exception:
        pass

    # 可选：仅做一次 pos_index 自检（默认关闭，设置 CHECK_POS_INDEX_ONCE=1 开启）
    do_pos_check_once = (os.environ.get('CHECK_POS_INDEX_ONCE', '0') == '1')
    pos_check_done = False

    for epoch in range(start_epoch, config['pretrain_epochs']):
        # 分布式Sampler设定epoch，确保每轮shuffle不同
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            try:
                train_loader.sampler.set_epoch(epoch)
            except Exception:
                pass
        # 训练
        model.train()
        gpu_loss_sum = torch.zeros(1, device=device)  # 在GPU上累计，避免每步同步
        last_loss_display = 0.0
        last_flux_rmse = 0.0
        train_flux_rmse = 0.0  # 累计训练RMSE以便epoch平均
        num_batches = 0
        # 验证频率（环境变量优先生效）
        val_frequency = int(os.environ.get('VAL_FREQUENCY', str(config.get('val_frequency', 200))))
        # 步级验证是否遍历完整的小验证集（1500光谱）。
        # 默认按少量批次快速验证，可通过环境变量 VAL_STEP_USE_FULL=1 或配置 val_step_use_full=True 启用全量验证。
        val_step_use_full = (os.environ.get('VAL_STEP_USE_FULL', '0') == '1') or bool(config.get('val_step_use_full', False))
        # 验证RMSE批次数（仅轻量模式下使用；环境变量优先生效）
        env_val_rmse_batches = int(os.environ.get('VAL_RMSE_BATCHES', str(config.get('val_rmse_batches', 4))))
        
        progress_bar = tqdm(
            train_loader,
            desc=f"预训练 Epoch {epoch+1}",
            disable=not (is_main_process and config.get('enable_progress', False))
        )
        # 首批同步：避免某些rank因首批IO慢导致其它rank提前进入all_reduce而超时
        first_batch_barrier_done = False

        for batch_idx, batch in enumerate(progress_bar):
            input_ids_cpu = batch['sequence']  # 保持在CPU，避免一次性将整批搬到GPU

            # 单步断言：仅主进程、仅一次、仅在首个训练批执行
            if (not pos_check_done) and do_pos_check_once and is_main_process:
                try:
                    pos_index_dbg = batch.get('pos_index', None)
                    if pos_index_dbg is None:
                        logger.warning("[pos_index] 未在batch中找到，跳过自检")
                    else:
                        # dtype/范围/形状检查
                        assert pos_index_dbg.dtype == torch.long, f"pos_index.dtype={pos_index_dbg.dtype} 不是 torch.long"
                        assert pos_index_dbg.min().item() >= 0, "pos_index 存在小于0的值"
                        assert pos_index_dbg.max().item() <= (config['seq_len'] - 1), "pos_index 超过 block_size-1"
                        assert tuple(pos_index_dbg.shape) == tuple(input_ids_cpu.shape), "pos_index.shape 与 input_ids 不一致"
                        # 取第一个样本打印前50与最大值
                        _p0 = pos_index_dbg[0, :50].tolist()
                        _pmax = int(pos_index_dbg.max().item())
                        logger.info(f"[pos_index] 形状={tuple(pos_index_dbg.shape)}  前50={_p0}  max={_pmax}")
                except Exception as e:
                    logger.warning(f"[pos_index] 自检失败: {e}")
                finally:
                    pos_check_done = True

            if is_distributed and not first_batch_barrier_done:
                try:
                    dist.barrier()
                except Exception:
                    pass
                first_batch_barrier_done = True

            # 初始化/自适应微批大小
            if dynamic_micro_bs is None:
                # 支持通过环境变量设置初始微批，避免首批OOM反复回退
                _env_init_mb = os.environ.get('INIT_MICRO_BATCH', '').strip()
                if _env_init_mb:
                    try:
                        dynamic_micro_bs = max(1, int(_env_init_mb))
                    except Exception:
                        dynamic_micro_bs = int(input_ids_cpu.size(0))
                else:
                    dynamic_micro_bs = int(input_ids_cpu.size(0))  # 先尝试整批
                # 若断点中包含微批大小，采用之
                if 'dynamic_micro_bs' in resume_misc:
                    try:
                        dynamic_micro_bs = int(resume_misc['dynamic_micro_bs'])
                    except Exception:
                        pass
            tried_reduce = False
            while True:
                try:
                    num_samples = int(input_ids_cpu.size(0))
                    micro_bs = max(1, min(dynamic_micro_bs, num_samples))
                    num_micro = (num_samples + micro_bs - 1) // micro_bs

                    # 训练一个大batch前先清零梯度
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_sum = 0.0

                    # 本批使用的 Random_CL 概率（线性调度）
                    rand_ratio_now = _current_cl_ratio()

                    for m in range(num_micro):
                        start = m * micro_bs
                        end = min(num_samples, start + micro_bs)
                        input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)

                        # 除最后一个微批外关闭梯度同步，降低通信/内存
                        no_sync_ctx = model.no_sync() if isinstance(model, DDP) and (m < num_micro - 1) else nullcontext()
                        with no_sync_ctx:
                            autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                            with autocast_ctx:
                                pos_index = batch.get('pos_index', None)
                                if pos_index is not None:
                                    pos_index = pos_index[start:end].to(device, non_blocking=True)
                                logits, loss = model(input_ids, mode='Random_CL', random_ratio=rand_ratio_now, loss_fn=focal_loss, pos_index=pos_index)
                                loss = loss / num_micro  # 平均化到一个大batch
                            loss.backward()
                            batch_loss_sum += loss.detach().item()

                        # 仅在需要时计算一次RMSE以减少显存/CPU压力
                        if is_main_process and (m == 0) and (batch_idx == 0 or ((batch_idx + 1) % metric_interval == 0)):
                            flux_rmse = flux_calc.calculate(logits, input_ids)
                            last_flux_rmse = flux_rmse
                        else:
                            flux_rmse = last_flux_rmse

                    # 一个大batch结束，做一次step和lr调度（保持原语义：每DataLoader批次一步）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    current_lr = scheduler.step()
                    global_step += 1  # 更新全局步数

                    # 累计训练loss/RMSE
                    gpu_loss_sum += torch.as_tensor(batch_loss_sum, device=device)
                    train_flux_rmse += flux_rmse
                    num_batches += 1

                    # 每N步保存检查点（可关闭）
                    if enable_step_ckpt and is_main_process and global_step % step_checkpoint_interval == 0:
                        logger.info(f"💾 保存步数检查点: step_{global_step}")
                        save_step_checkpoint(
                            global_step=global_step,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            config=config,
                            best_val_loss=best_val_loss,
                            early_stopping_state={
                                'best_loss': early_stopping.best_loss,
                                'counter': early_stopping.counter,
                            },
                            extra={'dynamic_micro_bs': dynamic_micro_bs},
                            keep_last_n=keep_n_checkpoints
                        )
                    
                    # 每N步进行验证
                    if global_step % val_frequency == 0 and global_step > 0:
                        logger.info(f"🔍 Step {global_step}: 开始验证(子集)...")
                        model.eval()
                        step_val_flux_rmse = 0.0
                        step_val_loss_random_cl = 0.0
                        step_val_batches = 0
                        val_rmse_batches_limit = min(10, env_val_rmse_batches)  # 限制验证批次数
                        
                        with torch.no_grad():
                            if val_step_use_full:
                                # 遍历完整的小验证集，保证每次评估基准一致
                                rand_ratio_eval = _current_cl_ratio()
                                for batch in val_loader:
                                    input_ids_cpu = batch['sequence']
                                    num_samples = int(input_ids_cpu.size(0))
                                    micro_bs = max(1, dynamic_micro_bs or num_samples)
                                    num_micro = (num_samples + micro_bs - 1) // micro_bs
                                    batch_val_loss_sum = 0.0
                                    logits_ref = None
                                    first_mb_ids = None
                                    for m in range(num_micro):
                                        start = m * micro_bs
                                        end = min(num_samples, start + micro_bs)
                                        input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)
                                        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                                        with autocast_ctx:
                                            pos_index = batch.get('pos_index', None)
                                            if pos_index is not None:
                                                pos_index = pos_index[start:end].to(device, non_blocking=True)
                                            _logits_cl, loss_cl = model(input_ids, mode='Random_CL', random_ratio=rand_ratio_eval, loss_fn=focal_loss, pos_index=pos_index)
                                        batch_val_loss_sum += (loss_cl.item() / num_micro)
                                        if logits_ref is None:
                                            logits_ref = _logits_cl
                                            first_mb_ids = input_ids
                                    if logits_ref is not None and first_mb_ids is not None:
                                        # 评估用 AR 顺序，得到更真实的RMSE
                                        try:
                                            num_samples = int(input_ids_cpu.size(0))
                                            micro_bs_eval = max(1, dynamic_micro_bs or num_samples)
                                            pos_m0 = batch.get('pos_index', None)
                                            if pos_m0 is not None:
                                                pos_m0 = pos_m0[:micro_bs_eval].to(device, non_blocking=True)
                                            autocast_ctx2 = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                                            with autocast_ctx2:
                                                logits_ar, _ = model(first_mb_ids, mode='AR', pos_index=pos_m0)
                                            flux_rmse = flux_calc.calculate(logits_ar, first_mb_ids)
                                        except Exception:
                                            # 回退：若AR评估失败，使用Random_CL的参考值
                                            flux_rmse = flux_calc.calculate(logits_ref, first_mb_ids)
                                        step_val_flux_rmse += flux_rmse
                                    step_val_loss_random_cl += batch_val_loss_sum
                                    step_val_batches += 1
                            else:
                                # 轻量评估：仅取前K个批次
                                val_iter = iter(val_loader)
                                rand_ratio_eval = _current_cl_ratio()
                                for val_batch_idx in range(val_rmse_batches_limit):
                                    try:
                                        batch = next(val_iter)
                                        input_ids_cpu = batch['sequence']
                                        num_samples = int(input_ids_cpu.size(0))
                                        micro_bs = max(1, dynamic_micro_bs or num_samples)
                                        num_micro = (num_samples + micro_bs - 1) // micro_bs
                                        batch_val_loss_sum = 0.0
                                        logits_ref = None
                                        first_mb_ids = None
                                        for m in range(num_micro):
                                            start = m * micro_bs
                                            end = min(num_samples, start + micro_bs)
                                            input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)
                                            autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                                        with autocast_ctx:
                                            pos_index = batch.get('pos_index', None)
                                            if pos_index is not None:
                                                pos_index = pos_index[start:end].to(device, non_blocking=True)
                                            _logits_cl, loss_cl = model(input_ids, mode='Random_CL', random_ratio=rand_ratio_eval, loss_fn=focal_loss, pos_index=pos_index)
                                            batch_val_loss_sum += (loss_cl.item() / num_micro)
                                            if logits_ref is None:
                                                logits_ref = _logits_cl
                                                first_mb_ids = input_ids
                                        if logits_ref is not None and first_mb_ids is not None:
                                            # 评估用 AR 顺序
                                            try:
                                                num_samples = int(input_ids_cpu.size(0))
                                                micro_bs_eval = max(1, dynamic_micro_bs or num_samples)
                                                pos_m0 = batch.get('pos_index', None)
                                                if pos_m0 is not None:
                                                    pos_m0 = pos_m0[:micro_bs_eval].to(device, non_blocking=True)
                                                autocast_ctx2 = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                                                with autocast_ctx2:
                                                    logits_ar, _ = model(first_mb_ids, mode='AR', pos_index=pos_m0)
                                                flux_rmse = flux_calc.calculate(logits_ar, first_mb_ids)
                                            except Exception:
                                                flux_rmse = flux_calc.calculate(logits_ref, first_mb_ids)
                                            step_val_flux_rmse += flux_rmse
                                        step_val_loss_random_cl += batch_val_loss_sum
                                        step_val_batches += 1
                                    except StopIteration:
                                        break
                                    except Exception as e:
                                        logger.warning(f"验证批次 {val_batch_idx} 出错: {e}")
                                        continue
                        
                        # 计算平均验证指标
                        if step_val_batches > 0:
                            # 本rank均值
                            avg_step_val_flux_rmse = step_val_flux_rmse / step_val_batches
                            avg_step_val_loss_random_cl = step_val_loss_random_cl / max(1, step_val_batches)

                            # 全局聚合
                            if is_distributed:
                                agg = torch.as_tensor([
                                    step_val_loss_random_cl, float(step_val_batches),
                                    step_val_flux_rmse, float(step_val_batches)
                                ], device=device, dtype=torch.float32)
                                dist.all_reduce(agg, op=dist.ReduceOp.SUM)
                                total_randcl_sum, total_loss_batches, total_rmse_sum, total_rmse_batches = agg.tolist()
                                avg_step_val_loss_random_cl = total_randcl_sum / max(1.0, total_loss_batches)
                                avg_step_val_flux_rmse = total_rmse_sum / max(1.0, total_rmse_batches)
                            else:
                                avg_step_val_loss_random_cl = step_val_loss_random_cl / max(1, step_val_batches)

                            if is_main_process:
                                logger.info(f"📊 Step {global_step} 验证结果:")
                                logger.info(f"   验证损失(Random_CL): {avg_step_val_loss_random_cl:.4f}")
                                logger.info(f"   验证Flux RMSE: {avg_step_val_flux_rmse:.2f}")
                                
                                # 仅按RMSE更新最佳模型（统一命名为 best_pretrain_model_step_*.pth）
                                if avg_step_val_flux_rmse < best_val_rmse:
                                    best_val_rmse = avg_step_val_flux_rmse
                                    best_state = (model.module if isinstance(model, DDP) else model).state_dict()
                                    torch.save({
                                        'model_state_dict': best_state,
                                        'global_step': global_step,
                                        'val_loss': avg_step_val_loss_random_cl,
                                        'val_loss_random_cl': avg_step_val_loss_random_cl,
                                        'flux_rmse': avg_step_val_flux_rmse,
                                        'epoch': epoch,
                                        'config': config
                                    }, f'output/best_pretrain_model_step_{global_step}.pth')
                                    try:
                                        cleanup_best_checkpoints_by_metric(prefix='best_pretrain_model_step_', meta_key='flux_rmse', keep_k=keep_k_best_rmse)
                                    except Exception as e:
                                        logger.warning(f"清理best(RMSE)失败: {e}")
                        
                        model.train()  # 返回训练模式

                        # 保存一步级别的最新模型（覆盖写入），确保中途停止也有最近一次权重
                        if is_main_process:
                            try:
                                save_checkpoint(
                                    path=f'output/last_step_model.pth',
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    config=config,
                                    best_val_loss=early_stopping.best_loss,
                                    early_stopping_state={
                                        'best_loss': early_stopping.best_loss,
                                        'counter': early_stopping.counter,
                                    },
                                    extra={'dynamic_micro_bs': dynamic_micro_bs},
                                    current_step=global_step,
                                )
                            except Exception as e:
                                logger.warning(f"保存 last_step_model 失败: {e}")

                    # 进度显示
                    if is_main_process and (batch_idx == 0 or ((batch_idx + 1) % metric_interval == 0)):
                        last_loss_display = batch_loss_sum
                        progress_bar.set_postfix({
                            'loss': f'{last_loss_display:.4f}',
                            'flux_rmse': f'{flux_rmse:.2f}'
                        })
                    break  # 本批成功，跳出while
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        if dynamic_micro_bs > 1:
                            dynamic_micro_bs = max(1, dynamic_micro_bs // 2)
                            tried_reduce = True
                            if is_main_process:
                                logger.warning(f"CUDA OOM: 自动将微批大小降至 {dynamic_micro_bs} 重新尝试该批次")
                            continue  # 重新尝试该批
                    # 其他错误或已无法继续缩小微批，向外抛出
                    raise
            # 如果本批曾因OOM缩小过微批，后续批次沿用更小的设置
            if tried_reduce and is_main_process:
                logger.warning(f"微批已缩小稳定为 {dynamic_micro_bs}，将用于后续批次")
        
        # 轮末完整验证（使用全量验证集；若未提供则退回子集）
        model.eval()
        val_loss = 0.0  # 使用 Random_CL 验证损失
        val_flux_rmse = 0.0
        val_batches = 0
        val_rmse_batches_limit = env_val_rmse_batches
        rmse_count = 0
        
        with torch.no_grad():
            for batch in (val_loader_full if val_loader_full is not None else val_loader):
                input_ids_cpu = batch['sequence']
                num_samples = int(input_ids_cpu.size(0))
                micro_bs = max(1, dynamic_micro_bs or num_samples)
                num_micro = (num_samples + micro_bs - 1) // micro_bs
                batch_val_loss_sum = 0.0
                logits_ref = None
                first_mb_ids = None
                rand_ratio_eval = _current_cl_ratio()
                for m in range(num_micro):
                    start = m * micro_bs
                    end = min(num_samples, start + micro_bs)
                    input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)
                    autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                    with autocast_ctx:
                        pos_index = batch.get('pos_index', None)
                        if pos_index is not None:
                            pos_index = pos_index[start:end].to(device, non_blocking=True)
                        logits_cl, loss_cl = model(input_ids, mode='Random_CL', random_ratio=rand_ratio_eval, loss_fn=focal_loss, pos_index=pos_index)
                    batch_val_loss_sum += (loss_cl.item() / num_micro)
                    if logits_ref is None:
                        logits_ref = logits_cl
                        first_mb_ids = input_ids
                val_loss += batch_val_loss_sum
                val_batches += 1
                if (rmse_count < val_rmse_batches_limit) and logits_ref is not None and first_mb_ids is not None:
                    # 评估用 AR 顺序
                    try:
                        num_samples2 = int(input_ids_cpu.size(0))
                        micro_bs_eval2 = max(1, dynamic_micro_bs or num_samples2)
                        pos2 = batch.get('pos_index', None)
                        if pos2 is not None:
                            pos2 = pos2[:micro_bs_eval2].to(device, non_blocking=True)
                        autocast_ctx2 = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                        with autocast_ctx2:
                            logits_ar2, _ = model(first_mb_ids, mode='AR', pos_index=pos2)
                        flux_rmse2 = flux_calc.calculate(logits_ar2, first_mb_ids)
                    except Exception:
                        flux_rmse2 = flux_calc.calculate(logits_ref, first_mb_ids)
                    val_flux_rmse += float(flux_rmse2)
                    rmse_count += 1
                # 不再计算验证准确率
        
        # 汇总并规范化各进程度量
        if is_distributed:
            # 训练损失/批次数
            total_train_loss = gpu_loss_sum.clone()
            dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
            total_num_batches = torch.as_tensor([num_batches], device=device, dtype=torch.float32)
            dist.all_reduce(total_num_batches, op=dist.ReduceOp.SUM)
            avg_train_loss = (total_train_loss.item() / max(1.0, total_num_batches.item()))
            # 验证损失/批次数
            total_val_loss = torch.as_tensor([val_loss], device=device, dtype=torch.float32)
            total_val_batches = torch.as_tensor([val_batches], device=device, dtype=torch.float32)
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_batches, op=dist.ReduceOp.SUM)
            avg_val_loss = (total_val_loss.item() / max(1.0, total_val_batches.item()))
        else:
            avg_train_loss = gpu_loss_sum.item() / num_batches
            avg_val_loss = val_loss / val_batches
        # RMSE聚合（验证）
        avg_train_flux_rmse = train_flux_rmse / max(1, num_batches)
        if is_distributed:
            rmse_agg = torch.as_tensor([val_flux_rmse, float(rmse_count)], device=device, dtype=torch.float32)
            dist.all_reduce(rmse_agg, op=dist.ReduceOp.SUM)
            total_rmse_sum, total_rmse_count = rmse_agg.tolist()
            avg_val_flux_rmse = (total_rmse_sum / max(1.0, total_rmse_count))
        else:
            avg_val_flux_rmse = (val_flux_rmse / max(1, rmse_count))
        
        # 记录（简化：仅记录核心指标）
        if is_main_process:
            monitor.log_pretrain(
                epoch + 1, avg_train_loss, avg_val_loss,
                avg_train_flux_rmse, avg_val_flux_rmse
            )
            # 不再输出验证准确率
        # 记录最后一个epoch的验证损失
        last_epoch_val_loss = avg_val_loss
            
        # 保存断点（last与按epoch）
        if is_main_process:
            save_checkpoint(
                path=f'output/last_checkpoint.pth',
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                best_val_loss=early_stopping.best_loss,
                early_stopping_state={
                    'best_loss': early_stopping.best_loss,
                    'counter': early_stopping.counter,
                },
                extra={'dynamic_micro_bs': dynamic_micro_bs},
                current_step=global_step
            )
            # 不再按epoch落盘 checkpoint-epoch*.pth（仅保留last_checkpoint/step/best/last_step）

        # 早停检查
        stop_tensor = torch.zeros(1, device=device)
        if is_main_process:
            # 如果从断点恢复，恢复早停状态
            if 'early_stopping' in resume_misc and epoch == start_epoch:
                es = resume_misc['early_stopping']
                try:
                    early_stopping.best_loss = float(es.get('best_loss', early_stopping.best_loss))
                    early_stopping.counter = int(es.get('counter', early_stopping.counter))
                except Exception:
                    pass
            # 改为基于验证RMSE的早停（数值越小越好）
            stop = early_stopping(avg_val_flux_rmse, model)
            stop_tensor[0] = 1.0 if stop else 0.0
        if is_distributed:
            dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1.0:
            if is_main_process:
                logger.info(f"🛑 早停触发！在第 {epoch + 1} 轮停止训练")
                logger.info(f"   最佳验证RMSE: {early_stopping.best_loss:.4f}")
            # 同步主进程可能恢复的最佳权重到所有进程
            if is_distributed:
                for param in model.parameters():
                    dist.broadcast(param.data, src=0)
            break
            
        # 精简日志：去除重复的epoch汇总输出（保留monitor日志）
    
    return last_epoch_val_loss if last_epoch_val_loss is not None else best_val_loss

def main_impl():
    """主训练实现（可由单进程直接调用，也可由spawn子进程调用）"""
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # 保持默认多进程启动方式（由torchrun控制），避免在脚本内强行修改
    
    # 分布式初始化（支持torchrun或本脚本spawn）
    world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size_env > 1
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    global_rank = int(os.environ.get('RANK', '0'))
    is_main_process = (global_rank == 0)

    # 仅主进程输出启动提示，非主进程降低日志等级避免重复输出
    if is_main_process:
        print("🚀 简化但高效的全流程训练")
        print("=" * 50)
        print(f"📋 日志文件保存到: {log_file}")
        logger.info("🚀 开始训练流程")
    else:
        logger.setLevel(logging.WARNING)
        for h in logger.handlers:
            try:
                h.setLevel(logging.WARNING)
            except Exception:
                pass
    
    # 配置 - 使用完整数据集和高级训练技巧
    config = {
        'seq_len': 8192,
        'vocab_size': 147,
        'max_train_samples': None,
        'max_val_samples': None,
        # 大模型配置
        'n_embd': 896,
        'n_head': 14,
        'n_layer': 12,
        'cond_dim': 128,
        'batch_size': 128,
        'pretrain_epochs': int(os.environ.get('PRETRAIN_EPOCHS', '6')),
        'pretrain_lr': float(os.environ.get('PRETRAIN_LR', '2.6e-4')),
        'finetune_lr': 1.0e-4,
        'weight_decay': 0.1,
        'patience': int(os.environ.get('PATIENCE', '5')),
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'warmup_ratio': float(os.environ.get('WARMUP_RATIO', '0.05')),
        'min_lr_ratio': float(os.environ.get('MIN_LR_RATIO', '0.1')),
        'use_amp': True,
        'amp_dtype': 'bf16',
        # 控制台输出与训练节奏
        'train_metric_interval': 200,
        'enable_step_compare': False,
        'enable_plots': False,
        'enable_progress': True,
        'val_rmse_batches': 4,
        'val_frequency': 200,
        # epoch 级检查点策略
        'epoch_checkpoint_interval': 1,
        'keep_last_epoch_checkpoints': 5,
    }
    
    # 允许通过环境变量覆盖全局batch大小（便于DDP/显存调参）
    try:
        _env_bs = os.environ.get('BATCH_SIZE', '').strip()
        if _env_bs:
            config['batch_size'] = max(1, int(_env_bs))
            if is_main_process:
                logger.info(f"环境变量覆盖 batch_size={config['batch_size']}")
    except Exception:
        pass
      
    # 设备
    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size_env, rank=global_rank)
        else:
            device = torch.device('cuda')
    else:
        if is_distributed and not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size_env, rank=global_rank)
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    # 非主进程减少日志干扰（双保险）
    if not is_main_process:
        logger.setLevel(logging.WARNING)
    
    # 加载token映射
    token_to_id, id_to_token = load_token_mapping()
    # 同步词表大小，防止与配置不一致导致索引越界
    config['vocab_size'] = len(token_to_id)
    
    # 预处理器
    preprocessor = StreamlinedPreprocessor(token_to_id, id_to_token, config['seq_len'])
    
    # 预处理数据（支持流式模式，基于rank对spectrum_id分片）
    streaming_mode = os.environ.get('STREAMING_MODE', '0') == '1'
    # 在线分片与稳定哈希划分（避免落盘分片导致磁盘耗尽）
    on_the_fly_shard = os.environ.get('ON_THE_FLY_SHARDING', '0') == '1'
    hash_split_enable = os.environ.get('HASH_SPLIT_ENABLE', '0') == '1'
    split_mod_base = int(os.environ.get('HASH_SPLIT_BASE', '10'))
    split_threshold = int(os.environ.get('HASH_SPLIT_TRAIN_THRESHOLD', '9'))
    single_csv_path = os.environ.get('SINGLE_CSV_PATH', '').strip()
    # 若启用在线分片或单CSV哈希划分，则强制流式模式
    if on_the_fly_shard or single_csv_path:
        streaming_mode = True
    if is_main_process:
        logger.info("📊 预处理训练数据...")
    # 允许通过环境变量覆盖默认CSV路径
    train_csv = os.environ.get('TRAIN_CSV', '').strip()
    if not train_csv:
        raise FileNotFoundError("未设置 TRAIN_CSV 环境变量，请在启动脚本中导出 TRAIN_CSV=绝对路径")
    # 训练数据集
    if streaming_mode and (on_the_fly_shard or single_csv_path or hash_split_enable):
        # 在线分片/哈希划分：不进行磁盘分片，直接在读取时按rank与哈希筛选
        src_csv_for_train = single_csv_path if single_csv_path else train_csv
        train_dataset = StreamedSpectrumIterableDataset(
            src_csv_for_train,
            preprocessor,
            config['max_train_samples'],
            is_main_process,
            filter_by_rank=is_distributed,
            world_size=world_size_env,
            global_rank=global_rank,
            split_by_hash=hash_split_enable or bool(single_csv_path),
            split_mod_base=split_mod_base,
            split_threshold=split_threshold,
            is_train_split=True,
        )
        if is_main_process:
            logger.info("   使用在线分片/稳定哈希划分的流式训练数据集")
    else:
        # 沿用原逻辑（可选磁盘分片）
        if is_distributed:
            shard_dir_train = 'pretrain_data/shards_train'
            _build_shards_by_spectrum_id(train_csv, shard_dir_train, world_size_env, is_main_process, label='train')
            selected_csv = os.path.join(shard_dir_train, f'train_rank{global_rank}.csv')
            if not os.path.isfile(selected_csv):
                raise FileNotFoundError(f"未找到rank{global_rank}的训练分片: {selected_csv}")
            if streaming_mode:
                train_dataset = StreamedSpectrumIterableDataset(selected_csv, preprocessor, config['max_train_samples'], is_main_process)
                if is_main_process:
                    logger.info("   使用流式IterableDataset加载训练分片")
            else:
                train_df = pd.read_csv(selected_csv)
                if is_main_process:
                    logger.info(f"   当前rank加载的训练行数: {len(train_df)} 条")
                train_data = preprocessor.preprocess_fast(train_df, config['max_train_samples'], is_main_process=is_main_process)
                if is_main_process:
                    logger.info(f"   处理后训练数据: {len(train_data)} 条")
                train_dataset = StreamlinedDataset(train_data)
        else:
            if streaming_mode:
                train_dataset = StreamedSpectrumIterableDataset(train_csv, preprocessor, config['max_train_samples'], is_main_process)
                if is_main_process:
                    logger.info("   使用流式IterableDataset加载训练数据（单进程）")
            else:
                train_df = pd.read_csv(train_csv)
                if is_main_process:
                    logger.info(f"   当前rank加载的训练行数: {len(train_df)} 条")
                train_data = preprocessor.preprocess_fast(train_df, config['max_train_samples'], is_main_process=is_main_process)
                if is_main_process:
                    logger.info(f"   处理后训练数据: {len(train_data)} 条")
                train_dataset = StreamlinedDataset(train_data)
    
    if is_main_process:
        logger.info("📊 预处理验证数据...")
    # 允许通过环境变量覆盖默认CSV路径，支持验证子集
    val_csv = os.environ.get('VAL_CSV', '').strip()
    val_subset_csv = os.environ.get('VAL_SUBSET_CSV', '').strip()
    
    # 优先使用验证子集（用于训练中验证），如果不存在则使用完整验证集
    if val_subset_csv and os.path.exists(val_subset_csv):
        val_csv_to_use = val_subset_csv
        if is_main_process:
            logger.info(f"📊 使用验证子集: {val_subset_csv}")
    elif val_csv and os.path.exists(val_csv):
        val_csv_to_use = val_csv
        if is_main_process:
            logger.info(f"📊 使用完整验证集: {val_csv}")
    else:
        raise FileNotFoundError("未找到验证集文件，请检查 VAL_CSV 或 VAL_SUBSET_CSV 环境变量")
    # 验证数据集
    if streaming_mode and (on_the_fly_shard or single_csv_path or hash_split_enable):
        # 若提供单CSV，则验证集也从同一CSV里按哈希9:1抽取
        src_csv_for_val = single_csv_path if single_csv_path else val_csv_to_use
        val_dataset = StreamedSpectrumIterableDataset(
            src_csv_for_val,
            preprocessor,
            config['max_val_samples'],
            is_main_process,
            filter_by_rank=is_distributed,
            world_size=world_size_env,
            global_rank=global_rank,
            split_by_hash=hash_split_enable or bool(single_csv_path),
            split_mod_base=split_mod_base,
            split_threshold=split_threshold,
            is_train_split=False,
        )
        if is_main_process:
            logger.info("   使用在线分片/稳定哈希划分的流式验证数据集")
    else:
        if is_distributed:
            shard_dir_val = 'pretrain_data/shards_val'
            _build_shards_by_spectrum_id(val_csv_to_use, shard_dir_val, world_size_env, is_main_process, label='val')
            selected_csv_val = os.path.join(shard_dir_val, f'val_rank{global_rank}.csv')
            if not os.path.isfile(selected_csv_val):
                raise FileNotFoundError(f"未找到rank{global_rank}的验证分片: {selected_csv_val}")
            if streaming_mode:
                val_dataset = StreamedSpectrumIterableDataset(selected_csv_val, preprocessor, config['max_val_samples'], is_main_process)
                if is_main_process:
                    logger.info("   使用流式IterableDataset加载验证分片")
            else:
                val_df = pd.read_csv(selected_csv_val)
                if is_main_process:
                    logger.info(f"   当前rank加载的验证行数: {len(val_df)} 条")
                val_data = preprocessor.preprocess_fast(val_df, config['max_val_samples'], is_main_process=is_main_process)
                if is_main_process:
                    logger.info(f"   处理后验证数据: {len(val_data)} 条")
                val_dataset = StreamlinedDataset(val_data)
        else:
            if streaming_mode:
                val_dataset = StreamedSpectrumIterableDataset(val_csv_to_use, preprocessor, config['max_val_samples'], is_main_process)
                if is_main_process:
                    logger.info("   使用流式IterableDataset加载验证数据（单进程）")
            else:
                val_df = pd.read_csv(val_csv_to_use)
                if is_main_process:
                    logger.info(f"   当前rank加载的验证行数: {len(val_df)} 条")
                val_data = preprocessor.preprocess_fast(val_df, config['max_val_samples'], is_main_process=is_main_process)
                if is_main_process:
                    logger.info(f"   处理后验证数据: {len(val_data)} 条")
                val_dataset = StreamlinedDataset(val_data)
    
    # 构建双验证数据集：子集用于步级验证；完整集用于epoch末全量验证
    val_dataset_small = None
    val_dataset_full = None
    # 当 VAL_STEP_USE_FULL=1 时，强制子集走内存缓存（固定基准、避免反复IO）
    val_step_use_full_flag = (os.environ.get('VAL_STEP_USE_FULL', '0') == '1') or False
    try:
        if val_subset_csv and os.path.exists(val_subset_csv):
            if val_step_use_full_flag:
                # 强制走内存（一次性加载1500光谱，约数百MB），保证每200步全量评估不受IO影响
                if is_main_process:
                    logger.info("   VAL_STEP_USE_FULL=1: 子集将以内存Dataset缓存以提升步级验证速度")
                val_df_small = pd.read_csv(val_subset_csv)
                val_data_small = preprocessor.preprocess_fast(val_df_small, None, is_main_process=is_main_process)
                val_dataset_small = StreamlinedDataset(val_data_small)
            else:
                if streaming_mode:
                    val_dataset_small = StreamedSpectrumIterableDataset(
                        val_subset_csv, preprocessor, config['max_val_samples'], is_main_process,
                        filter_by_rank=is_distributed, world_size=world_size_env, global_rank=global_rank,
                        split_by_hash=False, is_train_split=False,
                    )
                else:
                    val_df_small = pd.read_csv(val_subset_csv)
                    val_data_small = preprocessor.preprocess_fast(val_df_small, config['max_val_samples'], is_main_process=is_main_process)
                    val_dataset_small = StreamlinedDataset(val_data_small)
        if val_csv and os.path.exists(val_csv):
            if streaming_mode:
                val_dataset_full = StreamedSpectrumIterableDataset(
                    val_csv, preprocessor, config['max_val_samples'], is_main_process,
                    filter_by_rank=is_distributed, world_size=world_size_env, global_rank=global_rank,
                    split_by_hash=False, is_train_split=False,
                )
            else:
                val_df_full = pd.read_csv(val_csv)
                val_data_full = preprocessor.preprocess_fast(val_df_full, config['max_val_samples'], is_main_process=is_main_process)
                val_dataset_full = StreamlinedDataset(val_data_full)
    except Exception as e:
        if is_main_process:
            logger.warning(f"构建双验证数据集时出错: {e}")
    # 兜底：若缺失其一，退回已有的数据集
    if val_dataset_small is None:
        val_dataset_small = val_dataset
    if val_dataset_full is None:
        val_dataset_full = val_dataset

    # 动态确定DataLoader并行度与预取
    cpu_cores = os.cpu_count() or 8
    if streaming_mode:
        # 流式模式：支持多worker并行，提高CPU预处理/IO吞吐
        stream_workers = int(os.environ.get('STREAM_WORKERS', '2'))
        train_workers = max(0, stream_workers)
        val_workers = max(0, min(stream_workers // 2, stream_workers))
        prefetch_train = int(os.environ.get('STREAM_PREFETCH', '2')) if train_workers > 0 else None
        prefetch_val = int(os.environ.get('STREAM_PREFETCH', '2')) if val_workers > 0 else None
        stream_persistent = os.environ.get('STREAM_PERSISTENT', '1') == '1'
        stream_pin_memory = os.environ.get('STREAM_PIN_MEMORY', '1') == '1'
    else:
        train_workers = max(2, min(8, int(cpu_cores * 0.75)))
        val_workers = max(2, min(train_workers // 2, 6))
        prefetch_train = 6 if config['batch_size'] >= 64 else 4
        prefetch_val = 4

    # 预分片后不再使用DistributedSampler，避免“再切一刀”；仅调整每卡batch
    if is_distributed and world_size_env > 0:
        per_device_batch_size = max(1, config['batch_size'] // world_size_env)
        if (config['batch_size'] % world_size_env) != 0 and is_main_process:
            logger.warning(
                f"全局batch_size={config['batch_size']}不能被world_size={world_size_env}整除，"
                f"使用每卡batch={per_device_batch_size}（向下取整），有效全局batch={per_device_batch_size * world_size_env}"
            )
        if streaming_mode:
            train_loader = DataLoader(
                train_dataset,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=train_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and train_workers > 0),
                prefetch_factor=prefetch_train if train_workers > 0 else None,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
            # 构建步级与完整验证的Loader
            val_loader_small = DataLoader(
                val_dataset_small,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
            val_loader_full = DataLoader(
                val_dataset_full,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=per_device_batch_size,
                shuffle=True,
                num_workers=train_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_train,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
            val_loader_small = DataLoader(
                val_dataset_small,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
            val_loader_full = DataLoader(
                val_dataset_full,
                batch_size=per_device_batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
    else:
        if streaming_mode:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=train_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and train_workers > 0),
                prefetch_factor=prefetch_train if train_workers > 0 else None,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
            val_loader_small = DataLoader(
                val_dataset_small,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
            val_loader_full = DataLoader(
                val_dataset_full,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=stream_pin_memory,
                persistent_workers=(stream_persistent and val_workers > 0),
                prefetch_factor=prefetch_val if val_workers > 0 else None,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=train_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_train,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
            val_loader_small = DataLoader(
                val_dataset_small,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
            val_loader_full = DataLoader(
                val_dataset_full,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch_val,
                drop_last=True,
            )
    
    if is_main_process:
        # IterableDataset 没有 len()，跳过批次统计
        try:
            train_batches_info = f"train_batches={len(train_loader)}"
        except TypeError:
            train_batches_info = "train_batches=unknown(streaming)"
        try:
            val_batches_info = f"val_batches={len(val_loader_small)} / full={len(val_loader_full)}"
        except TypeError:
            val_batches_info = "val_batches=unknown(streaming)"
        
        logger.info(f"📈 DataLoader: {train_batches_info} (workers={train_workers}, prefetch={prefetch_train}), "
                    f"{val_batches_info} (workers={val_workers}, prefetch={prefetch_val}), "
                    f"batch_size={config['batch_size']}")

    # DDP首批预热：各rank先拉取一个训练批次，减少首批IO差导致的all_reduce超时
    warmup_first_batch = os.environ.get('WARMUP_FIRST_BATCH', '1') == '1'
    if warmup_first_batch and is_distributed:
        try:
            if is_main_process:
                logger.info("🔁 DDP首批预热：各rank预取1个训练批次并barrier对齐")
            _train_iter_warm = iter(train_loader)
            _ = next(_train_iter_warm)
        except StopIteration:
            if is_main_process:
                logger.warning("训练数据不足以完成首批预热，跳过")
        except Exception as e:
            if is_main_process:
                logger.warning(f"首批预热遇到异常：{e}")
        finally:
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass

    # 可选：仅进行IO吞吐基准测试，不进入训练
    io_benchmark = os.environ.get('IO_BENCHMARK', '0') == '1'
    if io_benchmark:
        bench_batches = int(os.environ.get('IO_BENCHMARK_BATCHES', '200'))
        if is_main_process:
            logger.info(f"🔬 IO基准: 仅读取训练集 {bench_batches} 批进行吞吐评估")
        t0 = time.time()
        num = 0
        for num, _batch in enumerate(train_loader, start=1):
            if num >= bench_batches:
                break
        dt = time.time() - t0
        bps = num / max(1e-6, dt)
        if is_main_process:
            logger.info(f"🔬 IO基准完成: 读取批次={num}, 用时={dt:.1f}s, 每秒批次={bps:.2f}")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return
    
    # 创建模型
    model = SpectrumDiffusionModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['seq_len'],
        cond_dim=config['cond_dim'],
        bos_token_id=int(config.get('bos_token_id', 0)),
        eos_token_id=int(config.get('eos_token_id', 1)),
        pad_token_id=int(config.get('pad_token_id', 2)),
        mask_token_id=int(os.environ.get('MASK_TOKEN_ID', str(int(config.get('pad_token_id', 2)))))
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None,
                    find_unused_parameters=False)
    # 不启用torch.compile，以保持完全一致的步数/数值路径
    
    if is_main_process:
        logger.info(f"模型参数数量: {model.module.get_num_params():,}" if isinstance(model, DDP) else f"模型参数数量: {model.get_num_params():,}")
    
    # 创建计算器和监控器
    flux_calc = FluxRMSECalculator(token_to_id, id_to_token)
    monitor = TrainingMonitor(enable_plots=config.get('enable_plots', False))
    
    # 预训练阶段
    logger.info("\n🎯 第一阶段：预训练")
    pretrain_loss = pretrain_phase(model, train_loader, val_loader_small, val_loader_full, config, flux_calc, monitor)
    
    # 保存历史
    with open(f'output/training_history_{timestamp}.json', 'w') as f:
        json.dump(monitor.history, f, indent=2)
    
    if is_main_process:
        logger.info(f"\n🎉 训练完成！最后一个epoch的验证损失: {pretrain_loss:.4f}")
        logger.info(f"总训练时间: {time.time() - monitor.start_time:.1f}秒")
        logger.info("\n📁 输出文件位置:")
        logger.info(f"   📋 训练历史: output/training_history_{timestamp}.json")
        logger.info(f"   📝 日志文件: {log_file}")
    # 释放分布式资源
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _spawn_entry(local_rank: int, world_size: int):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)  # 单机多卡
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['SPAWNED_DDP'] = '1'
    main_impl()


def main():
    """入口：支持直接python运行，自动spawn多进程（无需torchrun）"""
    # 如果外部未提供WORLD_SIZE，且本机有多卡，则自动spawn
    # 由torchrun控制多进程启动方式，无需在脚本内修改
    world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    spawned = os.environ.get('SPAWNED_DDP', '0') == '1'
    if world_size_env == 1 and not spawned and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            world_size = num_gpus  # 使用所有可用GPU
            mp.spawn(_spawn_entry, nprocs=world_size, args=(world_size,))
            return
    # 单卡或已由外部/子进程设置好环境，直接运行
    main_impl()

if __name__ == "__main__":
    main()