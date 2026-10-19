"""
优化微调脚本 (修正版)
- 序列长度拉满到8192
- 正确解析Tokenized格式的参数
- 基于预训练模型进行参数预测微调
- 监控参数RMSE变化
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VOCAB_CSV = _REPO_ROOT / "vocab" / "vocabulary.csv"
_src = _REPO_ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np
import logging
import json
import os
import hashlib
from spectral_lm.model_architecture import SpectrumDiffusionModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
from contextlib import nullcontext
warnings.filterwarnings('ignore')
from glob import glob

# 设置matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置日志（从环境变量读取LOG_PATH）
_log_path = os.environ.get('LOG_PATH', 'optimized_finetune_v2.log')
try:
    _log_dirname = os.path.dirname(_log_path)
    if _log_dirname:
        os.makedirs(_log_dirname, exist_ok=True)
except Exception:
    pass
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _get_main_process_flag() -> bool:
    try:
        return int(os.environ.get('RANK', '0')) == 0
    except Exception:
        return True

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    try:
        patterns = [
            os.path.join(ckpt_dir, 'finetune_ckpt_*.pt'),
            os.path.join(ckpt_dir, 'checkpoint_step_*.pth'),
            os.path.join(ckpt_dir, 'checkpoint_epoch_*.pth'),
        ]
        files: list[str] = []
        for p in patterns:
            files.extend(glob(p))
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    except Exception:
        return None

def _save_checkpoint(model, optimizer, scheduler, epoch: int, global_update_steps: int,
                     best_val_loss: float, config: dict, amp_dtype: torch.dtype, ckpt_dir: str,
                     tag: str) -> str | None:
    if not _get_main_process_flag():
        return None
    try:
        _ensure_dir(ckpt_dir)
        module_ref = model.module if isinstance(model, DDP) else model
        state = {
            'model_state_dict': module_ref.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'epoch': int(epoch),
            'global_update_steps': int(global_update_steps),
            'best_val_loss': float(best_val_loss),
            'config': dict(config) if isinstance(config, dict) else {},
            'amp_dtype': str(amp_dtype),
        }
        compat = bool(config.get('ckpt_compat_pretrain_naming', False))
        filename = f"finetune_ckpt_{tag}.pt"
        # 兼容命名：checkpoint_step_XXXX.pth / checkpoint_epoch_XXXX.pth
        try:
            if compat:
                if tag.startswith('step'):
                    num = int(''.join([c for c in tag if c.isdigit()]))
                    filename = f"checkpoint_step_{num}.pth"
                elif tag.startswith('epoch'):
                    num = int(''.join([c for c in tag if c.isdigit()]))
                    filename = f"checkpoint_epoch_{num}.pth"
        except Exception:
            pass
        path = os.path.join(ckpt_dir, filename)
        torch.save(state, path)
        logger.info(f"💾 保存检查点: {path}")
        # 写一个最新软链接/标记文件
        try:
            latest_path = os.path.join(ckpt_dir, 'latest.pt')
            torch.save({'path': path}, latest_path)
        except Exception:
            pass
        return path
    except Exception as e:
        if _get_main_process_flag():
            logger.warning(f"保存检查点失败: {e}")
        return None

def _update_step_topk_best(model, score_loss: float, step: int, ckpt_dir: str, topk: int = 3) -> None:
    """在步级评估时，基于子集avg_val_loss维护Top-K最优权重（仅模型权重）。

    - 在 `ckpt_dir` 下保存文件：best_step_loss{loss}_step{step}.pth
    - 在 `ckpt_dir` 下维护索引：best_step_topk.json，按loss升序保留前K个，其余自动删除
    """
    if not _get_main_process_flag():
        return
    try:
        _ensure_dir(ckpt_dir)
        meta_path = os.path.join(ckpt_dir, 'best_step_topk.json')
        # 读取现有条目
        try:
            with open(meta_path, 'r') as f:
                entries = json.load(f)
        except Exception:
            entries = []

        # 保存当前候选
        filename = f"best_step_loss{score_loss:.6f}_step{int(step):08d}.pth"
        save_path = os.path.join(ckpt_dir, filename)
        module_ref = model.module if isinstance(model, DDP) else model
        torch.save(module_ref.state_dict(), save_path)

        entries.append({'path': save_path, 'loss': float(score_loss), 'step': int(step)})
        entries.sort(key=lambda x: x.get('loss', float('inf')))

        # 只保留Top-K，删除其余文件
        while len(entries) > int(topk):
            removed = entries.pop(-1)
            try:
                if isinstance(removed, dict) and 'path' in removed and os.path.exists(removed['path']):
                    os.remove(removed['path'])
            except Exception:
                pass

        # 回写索引
        try:
            with open(meta_path, 'w') as f:
                json.dump(entries, f, indent=2)
        except Exception:
            pass
    except Exception:
        # 安静失败，不影响训练主流程
        return

def _load_checkpoint(model, optimizer, scheduler, resume_path: str) -> dict | None:
    try:
        if resume_path is None or resume_path == '' or (not os.path.exists(resume_path)):
            return None
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        opt_loaded_ok = False
        sch_loaded_ok = False
        module_ref = model.module if isinstance(model, DDP) else model
        missing, unexpected = module_ref.load_state_dict(ckpt.get('model_state_dict', {}), strict=False)
        if _get_main_process_flag():
            if missing or unexpected:
                logger.warning(f"恢复模型权重：missing={len(missing)} unexpected={len(unexpected)}")
        if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
            # 尝试加载优化器；若失败，基于ckpt内的param_groups数目重建一次优化器再尝试
            opt_state = ckpt['optimizer_state_dict']
            # 诊断信息：分组数量与每组参数个数
            try:
                saved_groups_dbg = len(opt_state.get('param_groups', []))
                saved_groups_param_lens = [len(g.get('params', [])) for g in opt_state.get('param_groups', [])]
            except Exception:
                saved_groups_dbg = None
                saved_groups_param_lens = []
            try:
                cur_groups_dbg = len(getattr(optimizer, 'param_groups', []))
                cur_groups_param_lens = [len(g.get('params', [])) for g in getattr(optimizer, 'param_groups', [])]
            except Exception:
                cur_groups_dbg = None
                cur_groups_param_lens = []
            if _get_main_process_flag():
                logger.info(f"尝试恢复优化器：saved_groups={saved_groups_dbg} current_groups={cur_groups_dbg}")
                if saved_groups_param_lens:
                    logger.info(f"ckpt每组参数个数: {saved_groups_param_lens}")
                if cur_groups_param_lens:
                    logger.info(f"当前每组参数个数: {cur_groups_param_lens}")
            def _build_optimizer_by_groups(saved_groups: int):
                # saved_groups==1：单组；>=2：主干+回归头两组（回归头乘以倍率）
                head_names = ['regression_head', 'param_head', 'head']
                lr = optimizer.param_groups[0].get('lr', float(os.environ.get('LR', '1e-4')))
                weight_decay = optimizer.param_groups[0].get('weight_decay', 0.01)
                head_lr_mult = float(os.environ.get('HEAD_LR_MULT', '4.0'))
                params_backbone, params_head = [], []
                for n, p in (model.named_parameters() if optimizer is not None else module_ref.named_parameters()):
                    (params_head if any(hn in n for hn in head_names) else params_backbone).append(p)
                if saved_groups <= 1:
                    return torch.optim.AdamW(module_ref.parameters(), lr=lr, weight_decay=weight_decay)
                groups = []
                if len(params_backbone) > 0:
                    groups.append({'params': params_backbone, 'lr': lr, 'weight_decay': weight_decay})
                if len(params_head) > 0:
                    groups.append({'params': params_head, 'lr': lr * max(1.0, head_lr_mult), 'weight_decay': weight_decay})
                return torch.optim.AdamW(groups)
            try:
                optimizer.load_state_dict(opt_state)
                opt_loaded_ok = True
            except Exception:
                try:
                    saved_groups = len(opt_state.get('param_groups', [])) or 1
                except Exception:
                    saved_groups = 1
                # 重建优化器并重试
                try:
                    new_opt = _build_optimizer_by_groups(saved_groups)
                    optimizer.__dict__.update(new_opt.__dict__)
                    optimizer.load_state_dict(opt_state)
                    opt_loaded_ok = True
                except Exception:
                    if _get_main_process_flag():
                        logger.warning("优化器状态恢复失败，跳过")
        if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                sch_loaded_ok = True
            except Exception:
                if _get_main_process_flag():
                    logger.warning("调度器状态恢复失败，跳过")
        # 回传恢复结果标记，供上层做兜底处理
        try:
            ckpt['_resume_info'] = {
                'optimizer_loaded': bool(opt_loaded_ok),
                'scheduler_loaded': bool(sch_loaded_ok),
                'saved_opt_groups': int(len(ckpt.get('optimizer_state_dict', {}).get('param_groups', []))) if isinstance(ckpt.get('optimizer_state_dict', {}), dict) else None
            }
        except Exception:
            pass
        return ckpt
    except Exception as e:
        if _get_main_process_flag():
            logger.warning(f"加载检查点失败: {e}")
        return None

def _resolve_sharded_path(original_csv: str, role: str, world_size: int, global_rank: int) -> tuple[str, bool]:
    """若存在按 rank 预分片的 CSV，则返回该分片路径并标记 using_shards=True；否则回退到原始路径。

    环境变量（可选）：
      - TRAIN_SHARD_DIR / VAL_SHARD_DIR：分片目录
      - TRAIN_PREFIX / VAL_PREFIX：输出前缀（默认取原始文件名去扩展名）
    文件命名：{prefix}_rank_{RANK}.csv
    """
    try:
        if role not in {"train", "val"}:
            return original_csv, False
        base_name = os.path.splitext(os.path.basename(original_csv))[0]
        shard_dir_env = os.environ.get('TRAIN_SHARD_DIR' if role == 'train' else 'VAL_SHARD_DIR', '').strip()
        if shard_dir_env == '':
            return original_csv, False
        prefix_env = os.environ.get('TRAIN_PREFIX' if role == 'train' else 'VAL_PREFIX', '').strip()
        prefix = prefix_env if prefix_env else base_name
        shard_path = os.path.join(shard_dir_env, f"{prefix}_rank_{global_rank}.csv")
        if os.path.exists(shard_path):
            return shard_path, True
        else:
            return original_csv, False
    except Exception:
        return original_csv, False

def load_token_mapping():
    """加载token映射（VOCAB_PATH）"""
    vocab_path = os.environ.get("VOCAB_PATH", str(_DEFAULT_VOCAB_CSV))
    vocab_df = pd.read_csv(vocab_path)
    token_to_id = dict(zip(vocab_df['token'], vocab_df['token_id']))
    return token_to_id

def load_pretrained_model(seq_len=8192):
    """加载预训练模型并适配新序列长度"""
    logger.info("🔄 加载预训练模型...")
    ckpt_path = os.environ.get('PRETRAIN_CKPT_PATH', '/home/share/guofangkeda/wangcunshi/Spectrum/Spec/Spec/pth/checkpoint_step_21000.pth')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # 创建新模型，序列长度拉满
    model = SpectrumDiffusionModel(
        vocab_size=147, 
        n_embd=896,
        n_head=14,
        n_layer=12,
        block_size=seq_len,
        cond_dim=128
    )
    
    # 智能加载权重，忽略不匹配的层（如位置嵌入）
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的权重
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]
    if skipped_keys:
        logger.warning(f"权重加载时跳过以下键 (形状不匹配): {skipped_keys}")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    logger.info("✅ 预训练模型加载成功，并已适配新序列长度")
    return model

def parse_tokenized_parameters(row):
    """从数据行中解析tokenized参数"""
    try:
        # 重建 Teff
        teff_str = (
            row.get('Teff_tthou', 'T0_tthou')[1] +
            row.get('Teff_thu', 'T0_thu')[1] +
            row.get('Teff_hun', 'T0_hun')[1] +
            row.get('Teff_ten', 'T0_ten')[1] +
            row.get('Teff_one', 'T0_one')[1]
        )
        teff = float(teff_str)

        # 重建 logg
        logg_str = (
            row.get('logg_hun', 'L0_hun')[1] + "." +
            row.get('logg_ten', 'L0_ten')[1] +
            row.get('logg_one', 'L0_one')[1]
        )
        logg = float(logg_str)

        # 重建 Fe/H
        feh_str = (
            row.get('FeH_ten', 'F0_ten')[1] + "." +
            row.get('FeH_one', 'F0_one')[1]
        )
        feh_val = float(feh_str)
        feh_sign_token = row.get('FeH_sign', 'F_pos')
        feh = feh_val if feh_sign_token == 'F_pos' else -feh_val

        return [teff, logg, feh]
    except (ValueError, TypeError, IndexError):
        return None

def get_param_stats(data):
    """计算参数的均值和标准差"""
    all_params = np.array([item['params'] for item in data if item.get('params') is not None])
    mean = all_params.mean(axis=0)
    std = all_params.std(axis=0)
    logger.info(f"参数统计: Mean={mean}, Std={std}")
    return torch.tensor(mean, dtype=torch.float), torch.tensor(std, dtype=torch.float)

def _stable_mod_by_world(value, world_size: int) -> int:
    bs = str(value).encode('utf-8')
    h = hashlib.md5(bs).hexdigest()
    return int(h[:8], 16) % max(1, world_size)

class StreamedFinetuneIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_path: str, token_to_id: dict, seq_len: int,
                 param_mean: torch.Tensor | None, param_std: torch.Tensor | None,
                 is_main_process: bool,
                 filter_by_rank: bool = False, world_size: int = 1, global_rank: int = 0):
        super().__init__()
        self.csv_path = csv_path
        self.token_to_id = token_to_id
        self.seq_len = seq_len
        self.param_mean = param_mean
        self.param_std = param_std
        self.is_main_process = is_main_process
        self.filter_by_rank = bool(filter_by_rank)
        self.world_size = int(world_size)
        self.global_rank = int(global_rank)

    def __iter__(self):
        token_to_id = self.token_to_id
        seq_len = self.seq_len
        bos = token_to_id.get('<BOS>', 0)
        eos = token_to_id.get('<EOS>', 1)
        pad = token_to_id.get('<SEP>', 2)
        none_tok = token_to_id.get('[None]', 3)
        flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']

        # DataLoader worker info
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # chunksize & sorting
        env_chunk = os.environ.get('CSV_CHUNK_SIZE', '').strip()
        chunk_size = int(env_chunk) if env_chunk.isdigit() else 1_000_000
        assume_sorted = os.environ.get('ASSUME_SORTED', '0') == '1'

        reader = pd.read_csv(self.csv_path, chunksize=chunk_size)
        current_id = None
        current_rows = []
        include_flag = False

        for chunk in reader:
            have_cols = set(chunk.columns.tolist())
            need_cols = {'spectrum_id', 'pixel_idx', *flux_columns, 'Teff_tthou','Teff_thu','Teff_hun','Teff_ten','Teff_one',
                         'logg_hun','logg_ten','logg_one','FeH_ten','FeH_one','FeH_sign'}
            keep_cols = list(have_cols & need_cols)
            if len(keep_cols) != len(chunk.columns):
                chunk = chunk[keep_cols]
            # 不进行块内排序；按输入顺序流式聚合同一 spectrum_id

            for _, row in chunk.iterrows():
                sid = row['spectrum_id']
                if current_id is None:
                    current_id = sid
                    include_flag = self._should_include_sid(sid, num_workers=num_workers, worker_id=worker_id)
                if sid != current_id:
                    if include_flag and current_rows:
                        yield from self._emit_sample(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)
                    current_id = sid
                    current_rows = []
                    include_flag = self._should_include_sid(sid, num_workers=num_workers, worker_id=worker_id)
                if include_flag:
                    current_rows.append(row)

        if current_id is not None and current_rows and include_flag:
            yield from self._emit_sample(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)

    def _should_include_sid(self, spectrum_id_value, num_workers: int = 1, worker_id: int = 0) -> bool:
        if self.filter_by_rank and self.world_size > 1:
            if _stable_mod_by_world(spectrum_id_value, self.world_size) != (self.global_rank % self.world_size):
                return False
        if num_workers > 1:
            if _stable_mod_by_world(spectrum_id_value, num_workers) != (worker_id % num_workers):
                return False
        return True

    def _emit_sample(self, spectrum_id, rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len):
        try:
            df = pd.DataFrame(rows)
            # 若存在像素列则按 pixel_idx 升序（与预训练对齐）
            if 'pixel_idx' in df.columns:
                try:
                    df = df.sort_values('pixel_idx')
                except Exception:
                    pass
            flux_matrix = df[flux_columns].values if all(col in df.columns for col in flux_columns) else np.empty((0,4), dtype=object)
            flux_tokens = []
            for row in flux_matrix:
                for token in row:
                    flux_tokens.append(token_to_id.get(token, none_tok))
            sequence = [bos] + flux_tokens + [eos]
            if len(sequence) > seq_len:
                sequence = sequence[:seq_len]
            else:
                sequence.extend([pad] * (seq_len - len(sequence)))
            # params
            params = parse_tokenized_parameters(df.iloc[0])
            if params is None:
                return
            params_tensor = torch.as_tensor(params, dtype=torch.float32)
            if (self.param_mean is not None) and (self.param_std is not None):
                std = self.param_std.clone()
                std[std == 0] = 1.0
                params_tensor = (params_tensor - self.param_mean) / std
            # 绝对像素位置索引：每像素重复4次，与序列对齐；BOS/EOS=0；可选阈值裁剪
            try:
                pos_max_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                pos_max_pixel = int(pos_max_env) if pos_max_env else None
            except Exception:
                pos_max_pixel = None
            abs_tokens = []
            if 'pixel_idx' in df.columns:
                try:
                    pix_list = [int(p) for p in df['pixel_idx'].tolist()]
                except Exception:
                    pix_list = []
                if pos_max_pixel is not None:
                    pix_list = [p for p in pix_list if p <= pos_max_pixel]
                for p in pix_list:
                    abs_tokens.extend([p] * 4)
            body_len = max(0, seq_len - 2)
            body_actual = min(body_len, max(0, len(sequence) - 2), len(abs_tokens))
            pos_index = [0] + abs_tokens[:body_actual] + [0]
            if len(pos_index) < seq_len:
                pos_index.extend([0] * (seq_len - len(pos_index)))
            sample = {
                'sequence': torch.as_tensor(sequence, dtype=torch.long),
                'params': params_tensor,
                'pos_index': torch.as_tensor(pos_index, dtype=torch.long)
            }
            yield sample
        except Exception:
            return

class OnlineDispatchIterableDataset(torch.utils.data.IterableDataset):
    """在线协调读取并分发（仅分布式使用）：
    - rank0：单进程流式读取 CSV，按“轮转分配”把每个完整光谱分配到各 rank；
      对于分配给 rank0 自身的数据，直接在本迭代器 yield；对于其他 rank，使用 dist.send 发送。
    - 其他 rank：阻塞式接收 dist.recv，按接收顺序 yield。
    - 保证：不拆分光谱；各 rank 内部样本顺序与原始 CSV 出现顺序一致；跨 rank 采用严格轮转保证步对齐。
    注意：要求 DataLoader num_workers=0。
    """
    def __init__(self, csv_path: str, token_to_id: dict, seq_len: int,
                 param_mean: torch.Tensor, param_std: torch.Tensor,
                 world_size: int, global_rank: int):
        super().__init__()
        self.csv_path = csv_path
        self.token_to_id = token_to_id
        self.seq_len = int(seq_len)
        self.param_mean = param_mean.clone().detach().to('cpu')
        self.param_std = param_std.clone().detach().to('cpu')
        self.world_size = int(world_size)
        self.global_rank = int(global_rank)
        # 通信设备：NCCL 仅支持 CUDA 张量；若可用则使用本地 GPU
        if torch.cuda.is_available():
            try:
                _lr = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
            except Exception:
                _lr = 0
            self.device = torch.device(f"cuda:{_lr}")
        else:
            self.device = torch.device('cpu')

    def __iter__(self):
        if not (dist.is_available() and dist.is_initialized()) or self.world_size <= 1:
            # 回退为本地流式
            yield from self._local_stream_iter()
            return
        if self.global_rank == 0:
            yield from self._coordinator_iter()
        else:
            yield from self._receiver_iter()

    # 本地回退：直接使用 StreamedFinetuneIterableDataset 的实现思路
    def _local_stream_iter(self):
        token_to_id = self.token_to_id
        seq_len = self.seq_len
        bos = token_to_id.get('<BOS>', 0)
        eos = token_to_id.get('<EOS>', 1)
        pad = token_to_id.get('<SEP>', 2)
        none_tok = token_to_id.get('[None]', 3)
        flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']
        env_chunk = os.environ.get('CSV_CHUNK_SIZE', '').strip()
        chunk_size = int(env_chunk) if env_chunk.isdigit() else 1_000_000
        reader = pd.read_csv(self.csv_path, chunksize=chunk_size)
        current_id = None
        current_rows = []
        for chunk in reader:
            have_cols = set(chunk.columns.tolist())
            need_cols = {'spectrum_id', 'pixel_idx', *flux_columns, 'Teff_tthou','Teff_thu','Teff_hun','Teff_ten','Teff_one',
                         'logg_hun','logg_ten','logg_one','FeH_ten','FeH_one','FeH_sign'}
            keep_cols = list(have_cols & need_cols)
            if len(keep_cols) != len(chunk.columns):
                chunk = chunk[keep_cols]
            for _, row in chunk.iterrows():
                sid = row['spectrum_id']
                if current_id is None:
                    current_id = sid
                if sid != current_id:
                    yield from self._build_and_yield(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)
                    current_id = sid
                    current_rows = []
                current_rows.append(row)
        if current_id is not None and current_rows:
            yield from self._build_and_yield(current_id, current_rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len)

    def _build_and_yield(self, spectrum_id, rows, flux_columns, token_to_id, none_tok, bos, eos, pad, seq_len):
        try:
            df = pd.DataFrame(rows)
            if 'pixel_idx' in df.columns:
                try:
                    df = df.sort_values('pixel_idx')
                except Exception:
                    pass
            flux_matrix = df[flux_columns].values if all(col in df.columns for col in flux_columns) else np.empty((0,4), dtype=object)
            flux_tokens = []
            for row in flux_matrix:
                for token in row:
                    flux_tokens.append(token_to_id.get(token, none_tok))
            sequence = [bos] + flux_tokens + [eos]
            if len(sequence) > seq_len:
                sequence = sequence[:seq_len]
            else:
                sequence.extend([pad] * (seq_len - len(sequence)))
            params = parse_tokenized_parameters(df.iloc[0])
            if params is None:
                return
            params_tensor = torch.as_tensor(params, dtype=torch.float32)
            std = self.param_std.clone()
            std[std == 0] = 1.0
            params_tensor = (params_tensor - self.param_mean) / std
            # 绝对位置索引
            try:
                pos_max_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                pos_max_pixel = int(pos_max_env) if pos_max_env else None
            except Exception:
                pos_max_pixel = None
            abs_tokens = []
            if 'pixel_idx' in df.columns:
                try:
                    pix_list = [int(p) for p in df['pixel_idx'].tolist()]
                except Exception:
                    pix_list = []
                if pos_max_pixel is not None:
                    pix_list = [p for p in pix_list if p <= pos_max_pixel]
                for p in pix_list:
                    abs_tokens.extend([p] * 4)
            body_len = max(0, seq_len - 2)
            body_actual = min(body_len, max(0, len(sequence) - 2), len(abs_tokens))
            pos_index = [0] + abs_tokens[:body_actual] + [0]
            if len(pos_index) < seq_len:
                pos_index.extend([0] * (seq_len - len(pos_index)))
            yield {
                'sequence': torch.as_tensor(sequence, dtype=torch.long),
                'params': params_tensor,
                'pos_index': torch.as_tensor(pos_index, dtype=torch.long)
            }
        except Exception:
            return

    def _coordinator_iter(self):
        token_to_id = self.token_to_id
        seq_len = self.seq_len
        bos = token_to_id.get('<BOS>', 0)
        eos = token_to_id.get('<EOS>', 1)
        pad = token_to_id.get('<SEP>', 2)
        none_tok = token_to_id.get('[None]', 3)
        flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']
        env_chunk = os.environ.get('CSV_CHUNK_SIZE', '').strip()
        chunk_size = int(env_chunk) if env_chunk.isdigit() else 1_000_000

        # 让所有 rank 都进入等待
        try:
            dist.barrier()
        except Exception:
            pass

        # 预热NCCL点对点通信（建立recv/send的communicator，避免其他rank首次recv超时）
        try:
            warm = torch.tensor([1234], dtype=torch.int32, device=self.device)
            for r in range(1, self.world_size):
                dist.send(warm, dst=r)
        except Exception:
            pass

        reader = pd.read_csv(self.csv_path, chunksize=chunk_size)
        current_id = None
        current_rows = []
        next_dest = 0
        # 启动阶段：为每个远端rank预取若干样本，避免其首次recv长时间等待
        prefetch_per_rank = max(1, int(os.environ.get('DISPATCH_PREFETCH_PER_RANK', '2')))
        delivered: dict[int, int] = {r: 0 for r in range(1, self.world_size)}
        bootstrap_done = (self.world_size <= 1)

        def pick_dest() -> int:
            nonlocal next_dest, bootstrap_done
            if not bootstrap_done:
                for r in range(1, self.world_size):
                    if delivered.get(r, 0) < prefetch_per_rank:
                        return r
                bootstrap_done = True
            dest = next_dest
            next_dest = (next_dest + 1) % self.world_size
            return dest

        def send_sample(dest_rank: int, seq_tensor: torch.Tensor, params_tensor: torch.Tensor):
            # 使用 CUDA 张量与 NCCL 兼容
            ctrl = torch.tensor([1], dtype=torch.int32, device=self.device)
            dist.send(ctrl, dst=dest_rank)
            dist.send(seq_tensor.to(self.device, non_blocking=True), dst=dest_rank)
            dist.send(params_tensor.to(self.device, non_blocking=True), dst=dest_rank)

        for chunk in reader:
            have_cols = set(chunk.columns.tolist())
            need_cols = {'spectrum_id', 'pixel_idx', *flux_columns, 'Teff_tthou','Teff_thu','Teff_hun','Teff_ten','Teff_one',
                         'logg_hun','logg_ten','logg_one','FeH_ten','FeH_one','FeH_sign'}
            keep_cols = list(have_cols & need_cols)
            if len(keep_cols) != len(chunk.columns):
                chunk = chunk[keep_cols]
            for _, row in chunk.iterrows():
                sid = row['spectrum_id']
                if current_id is None:
                    current_id = sid
                if sid != current_id:
                    # 完成一个光谱，构建样本
                    df = pd.DataFrame(current_rows)
                    if 'pixel_idx' in df.columns:
                        try:
                            df = df.sort_values('pixel_idx')
                        except Exception:
                            pass
                    flux_matrix = df[flux_columns].values if all(col in df.columns for col in flux_columns) else np.empty((0,4), dtype=object)
                    flux_tokens = []
                    for r in flux_matrix:
                        for token in r:
                            flux_tokens.append(token_to_id.get(token, none_tok))
                    sequence = [bos] + flux_tokens + [eos]
                    if len(sequence) > seq_len:
                        sequence = sequence[:seq_len]
                    else:
                        sequence.extend([pad] * (seq_len - len(sequence)))
                    params = parse_tokenized_parameters(df.iloc[0])
                    if params is not None:
                        params_tensor = torch.as_tensor(params, dtype=torch.float32)
                        std = self.param_std.clone()
                        std[std == 0] = 1.0
                        params_tensor = (params_tensor - self.param_mean) / std
                        # 绝对位置索引
                        try:
                            pos_max_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                            pos_max_pixel = int(pos_max_env) if pos_max_env else None
                        except Exception:
                            pos_max_pixel = None
                        abs_tokens = []
                        if 'pixel_idx' in df.columns:
                            try:
                                pix_list = [int(p) for p in df['pixel_idx'].tolist()]
                            except Exception:
                                pix_list = []
                            if pos_max_pixel is not None:
                                pix_list = [p for p in pix_list if p <= pos_max_pixel]
                            for p in pix_list:
                                abs_tokens.extend([p] * 4)
                        body_len = max(0, seq_len - 2)
                        body_actual = min(body_len, max(0, len(sequence) - 2), len(abs_tokens))
                        pos_index = [0] + abs_tokens[:body_actual] + [0]
                        if len(pos_index) < seq_len:
                            pos_index.extend([0] * (seq_len - len(pos_index)))
                        seq_tensor = torch.as_tensor(sequence, dtype=torch.long)
                        pos_tensor = torch.as_tensor(pos_index, dtype=torch.long)

                        dest = pick_dest()
                        if dest == 0:
                            yield {'sequence': seq_tensor, 'params': params_tensor, 'pos_index': pos_tensor}
                        else:
                            send_sample(dest, seq_tensor, params_tensor)
                            if not bootstrap_done and dest != 0:
                                delivered[dest] = delivered.get(dest, 0) + 1

                    current_id = sid
                    current_rows = []
                current_rows.append(row)

        # flush 最后一个光谱
        if current_id is not None and len(current_rows) > 0:
            df = pd.DataFrame(current_rows)
            if 'pixel_idx' in df.columns:
                try:
                    df = df.sort_values('pixel_idx')
                except Exception:
                    pass
            flux_matrix = df[flux_columns].values if all(col in df.columns for col in flux_columns) else np.empty((0,4), dtype=object)
            flux_tokens = []
            for r in flux_matrix:
                for token in r:
                    flux_tokens.append(token_to_id.get(token, none_tok))
            sequence = [bos] + flux_tokens + [eos]
            if len(sequence) > seq_len:
                sequence = sequence[:seq_len]
            else:
                sequence.extend([pad] * (seq_len - len(sequence)))
            params = parse_tokenized_parameters(df.iloc[0])
            if params is not None:
                params_tensor = torch.as_tensor(params, dtype=torch.float32)
                std = self.param_std.clone()
                std[std == 0] = 1.0
                params_tensor = (params_tensor - self.param_mean) / std
                # 构建 pos_index
                try:
                    pos_max_env = os.environ.get('POS_MAX_PIXEL', '').strip()
                    pos_max_pixel = int(pos_max_env) if pos_max_env else None
                except Exception:
                    pos_max_pixel = None
                abs_tokens = []
                if 'pixel_idx' in df.columns:
                    try:
                        pix_list = [int(p) for p in df['pixel_idx'].tolist()]
                    except Exception:
                        pix_list = []
                    if pos_max_pixel is not None:
                        pix_list = [p for p in pix_list if p <= pos_max_pixel]
                    for p in pix_list:
                        abs_tokens.extend([p] * 4)
                body_len = max(0, seq_len - 2)
                body_actual = min(body_len, max(0, len(sequence) - 2), len(abs_tokens))
                pos_index = [0] + abs_tokens[:body_actual] + [0]
                if len(pos_index) < seq_len:
                    pos_index.extend([0] * (seq_len - len(pos_index)))
                seq_tensor = torch.as_tensor(sequence, dtype=torch.long)
                pos_tensor = torch.as_tensor(pos_index, dtype=torch.long)
                dest = pick_dest()
                if dest == 0:
                    yield {'sequence': seq_tensor, 'params': params_tensor, 'pos_index': pos_tensor}
                else:
                    send_sample(dest, seq_tensor, params_tensor)
                    if not bootstrap_done and dest != 0:
                        delivered[dest] = delivered.get(dest, 0) + 1

        # 通知所有 rank 结束
        ctrl0 = torch.tensor([0], dtype=torch.int32, device=self.device)
        for r in range(1, self.world_size):
            try:
                dist.send(ctrl0, dst=r)
            except Exception:
                pass

    def _receiver_iter(self):
        seq_len = self.seq_len
        # barrier 确保协调器已就绪
        try:
            dist.barrier()
        except Exception:
            pass
        # 接收一次预热信号，提前创建NCCL点对点通信的communicator
        try:
            warm = torch.empty(1, dtype=torch.int32, device=self.device)
            dist.recv(warm, src=0)
        except Exception:
            pass
        while True:
            # 控制信号与数据均在 CUDA 上接收（NCCL 要求）
            ctrl = torch.empty(1, dtype=torch.int32, device=self.device)
            dist.recv(ctrl, src=0)
            if int(ctrl.item()) == 0:
                break
            seq_tensor = torch.empty(seq_len, dtype=torch.long, device=self.device)
            params_tensor = torch.empty(3, dtype=torch.float32, device=self.device)
            dist.recv(seq_tensor, src=0)
            dist.recv(params_tensor, src=0)
            # 直接返回 CUDA 张量；训练环节会 .to(device)（无额外拷贝）
            yield {'sequence': seq_tensor, 'params': params_tensor}

def compute_param_stats_streaming(csv_path: str, token_to_id: dict, seq_len: int, max_samples_for_stats: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """流式遍历前N个光谱估计参数均值方差，避免一次性加载全表。"""
    # 简单流式，只用于统计，不做rank/worker过滤，取前N个id
    env_chunk = os.environ.get('CSV_CHUNK_SIZE', '').strip()
    chunk_size = int(env_chunk) if env_chunk.isdigit() else 1_000_000
    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']
    collected = []
    current_id = None
    current_rows = []
    assume_sorted = os.environ.get('ASSUME_SORTED', '0') == '1'
    count_ids = 0
    for chunk in reader:
        have_cols = set(chunk.columns.tolist())
        need_cols = {'spectrum_id', 'pixel_idx', *flux_columns, 'Teff_tthou','Teff_thu','Teff_hun','Teff_ten','Teff_one',
                     'logg_hun','logg_ten','logg_one','FeH_ten','FeH_one','FeH_sign'}
        keep_cols = list(have_cols & need_cols)
        if len(keep_cols) != len(chunk.columns):
            chunk = chunk[keep_cols]
        # 不进行块内排序；按输入顺序流式聚合同一 spectrum_id
        for _, row in chunk.iterrows():
            sid = row['spectrum_id']
            if current_id is None:
                current_id = sid
            if sid != current_id:
                if current_rows:
                    df = pd.DataFrame(current_rows)
                    p = parse_tokenized_parameters(df.iloc[0])
                    if p is not None:
                        collected.append(p)
                        count_ids += 1
                        if count_ids >= max_samples_for_stats:
                            arr = np.array(collected, dtype=np.float32)
                            return torch.tensor(arr.mean(axis=0)), torch.tensor(arr.std(axis=0))
                current_id = sid
                current_rows = []
            current_rows.append(row)
    if current_rows:
        df = pd.DataFrame(current_rows)
        p = parse_tokenized_parameters(df.iloc[0])
        if p is not None:
            collected.append(p)
    if len(collected) == 0:
        mean = torch.tensor([5000.0, 4.5, 0.0], dtype=torch.float32)
        std = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    else:
        arr = np.array(collected, dtype=np.float32)
        mean = torch.tensor(arr.mean(axis=0))
        std = torch.tensor(arr.std(axis=0))
    return mean, std

def _load_param_stats_override_from_env() -> tuple[torch.Tensor, torch.Tensor] | None:
    """可选：从环境变量覆盖参数标准化 mean/std。

    支持两种方式：
    1) PARAM_STATS_JSON=/abs/path.json  文件内容: {"mean": [..,..,..], "std": [..,..,..]}
    2) PARAM_MEAN / PARAM_STD  逗号分隔字符串，例如 "5000,4.5,0"
    返回 torch.float32 张量；若无可用覆盖则返回 None。
    """
    try:
        json_path = os.environ.get('PARAM_STATS_JSON', '').strip()
        if json_path and os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                obj = json.load(f)
            mean_list = obj.get('mean', None)
            std_list = obj.get('std', None)
            names_list = obj.get('names', None)
            if isinstance(mean_list, (list, tuple)) and isinstance(std_list, (list, tuple)) and len(mean_list) == len(std_list) == 3:
                # 若提供了 names，则按 teff,logg,feh 顺序重排
                if isinstance(names_list, (list, tuple)) and len(names_list) == 3:
                    try:
                        # 归一化名称
                        norm = lambda s: str(s).strip().lower().replace(' ', '')
                        names_norm = [norm(n) for n in names_list]
                        # 接受 feh 或 fe/h
                        names_norm = ['feh' if n in ('fe/h','feh') else n for n in names_norm]
                        desired = ['teff', 'logg', 'feh']
                        idx_map = {names_norm[i]: i for i in range(3)}
                        if all(d in idx_map for d in desired):
                            mean_list = [mean_list[idx_map[d]] for d in desired]
                            std_list = [std_list[idx_map[d]] for d in desired]
                    except Exception:
                        pass
                m = torch.tensor([float(x) for x in mean_list], dtype=torch.float32)
                s = torch.tensor([float(x) for x in std_list], dtype=torch.float32)
                return m, s
        # 尝试解析环境变量 PARAM_MEAN / PARAM_STD
        mean_str = os.environ.get('PARAM_MEAN', '').strip()
        std_str = os.environ.get('PARAM_STD', '').strip()
        if mean_str and std_str:
            m_vals = [float(x) for x in mean_str.split(',')]
            s_vals = [float(x) for x in std_str.split(',')]
            if len(m_vals) == 3 and len(s_vals) == 3:
                m = torch.tensor(m_vals, dtype=torch.float32)
                s = torch.tensor(s_vals, dtype=torch.float32)
                return m, s
    except Exception:
        pass
    return None

def optimized_preprocess_data(csv_file, token_to_id, max_samples=None, seq_len=8192):
    """一次性读取CSV，按 spectrum_id（或存在 augmentation_id 时的复合键）聚合；
    为每条光谱按 pixel_idx 升序构造序列与绝对位置索引 pos_index（每像素重复4次，BOS/EOS=0）。
    """
    logger.info(f"📊 优化预处理(全量读入): {csv_file}, 序列长度: {seq_len}")

    df = pd.read_csv(csv_file)
    # 构造复合键（若有 augmentation_id）
    try:
        if 'augmentation_id' in df.columns:
            comp = df.apply(lambda r: f"{str(r.get('spectrum_id', r.get('obsid', '')))}|aug={str(int(r['augmentation_id'])) if str(r['augmentation_id']).isdigit() else str(r['augmentation_id'])}", axis=1)
            df = df.assign(comp_id=comp)
            grouped = df.groupby('comp_id')
        else:
            grouped = df.groupby('spectrum_id')
    except Exception:
        grouped = df.groupby('spectrum_id')

    items = list(grouped)
    if isinstance(max_samples, int) and max_samples > 0:
        items = items[:max_samples]

    results = []
    flux_columns = ['flux_thu', 'flux_hun', 'flux_ten', 'flux_one']

    bos_token = token_to_id.get('<BOS>', 0)
    eos_token = token_to_id.get('<EOS>', 1)
    pad_token = token_to_id.get('<SEP>', 2)
    none_token = token_to_id.get('[None]', 3)

    # 位置阈值（可选）
    try:
        pos_max_env = os.environ.get('POS_MAX_PIXEL', '').strip()
        pos_max_pixel = int(pos_max_env) if pos_max_env else None
    except Exception:
        pos_max_pixel = None

    for spectrum_id, group in tqdm(items, desc="优化预处理", disable=(not _get_main_process_flag())):
        # 排序像素
        if 'pixel_idx' in group.columns:
            try:
                group = group.sort_values('pixel_idx')
            except Exception:
                pass

        # flux -> token
        flux_matrix = group[flux_columns].values if all(col in group.columns for col in flux_columns) else np.empty((0,4), dtype=object)
        flux_tokens = []
        for row in flux_matrix:
            for token in row:
                flux_tokens.append(token_to_id.get(token, none_token))

        sequence = [bos_token] + flux_tokens + [eos_token]

        # 绝对位置：每像素重复4次（与 flux 四列一致），BOS/EOS=0
        abs_tokens = []
        if 'pixel_idx' in group.columns:
            try:
                pix_list = [int(p) for p in group['pixel_idx'].tolist()]
            except Exception:
                pix_list = []
            if pos_max_pixel is not None:
                pix_list = [p for p in pix_list if p <= pos_max_pixel]
            for p in pix_list:
                abs_tokens.extend([p] * 4)
        body_len = max(0, seq_len - 2)
        body_actual = min(body_len, max(0, len(sequence) - 2), len(abs_tokens))
        pos_index = [0] + abs_tokens[:body_actual] + [0]

        # 截断或填充
        if len(sequence) > seq_len:
            sequence = sequence[:seq_len]
        else:
            sequence.extend([pad_token] * (seq_len - len(sequence)))
        if len(pos_index) < seq_len:
            pos_index.extend([0] * (seq_len - len(pos_index)))

        # 参数
        params = parse_tokenized_parameters(group.iloc[0])
        if params is None:
            continue
        results.append({'sequence': sequence, 'params': params, 'pos_index': pos_index})

    logger.info(f"✅ 优化预处理完成(全量): {len(results)} 个有效样本")
    return results

class FinetuneDataset(Dataset):
    """微调数据集（带参数标准化）"""
    def __init__(self, data, param_mean, param_std):
        self.data = data
        self.param_mean = param_mean
        self.param_std = param_std
        # 避免除以零
        self.param_std[self.param_std == 0] = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 标准化参数
        normalized_params = (torch.tensor(item['params'], dtype=torch.float) - self.param_mean) / self.param_std
        result = {
            'sequence': torch.tensor(item['sequence'], dtype=torch.long),
            'params': normalized_params
        }
        if 'pos_index' in item and item['pos_index'] is not None:
            result['pos_index'] = torch.tensor(item['pos_index'], dtype=torch.long)
        return result

class ParameterRMSE:
    """参数RMSE计算器"""
    def __init__(self):
        self.param_names = ['Teff', 'logg', 'Fe/H']
    
    def calculate(self, pred, true, mean, std):
        """计算反标准化后的RMSE"""
        # 反标准化
        pred_unnorm = pred.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()
        true_unnorm = true.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()
        
        rmses = {}
        for i, name in enumerate(self.param_names):
            rmses[name] = np.sqrt(np.mean((pred_unnorm[:, i] - true_unnorm[:, i]) ** 2))
        rmses['total'] = np.sqrt(np.mean([v**2 for v in rmses.values()]))
        return rmses

def finetune(model, train_loader, val_loader, config, param_stats):
    """微调训练循环"""
    logger.info("🚀 开始微调训练...")
    is_distributed = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 冻结层（DDP 包裹后需访问 module）
    module_ref = model.module if isinstance(model, DDP) else model
    for i, block in enumerate(module_ref.blocks):
        if i < config['freeze_layers']:
            for param in block.parameters():
                param.requires_grad = False
    # 冻结与回归无关的头部与嵌入，减小显存
    for p in module_ref.lm_head.parameters():
        p.requires_grad = False
    for p in module_ref.wnonee.parameters():
        p.requires_grad = False
    # 绝对位置条件嵌入：默认不冻结，便于不同 block_size 的权重继续学习
    if os.environ.get('FREEZE_WTPE', '0') == '1':
        for p in module_ref.wtpe.parameters():
            p.requires_grad = False
    logger.info(f"冻结了前 {config['freeze_layers']} 层")
    
    # 预读 ckpt 决定优化器分组是否与 ckpt 对齐
    ckpt_dir = str(config.get('ckpt_dir', 'finetune_ckpts'))
    resume_from_cfg = str(config.get('resume_from', '') or '')
    auto_resume_latest_cfg = bool(config.get('auto_resume_latest', True))
    resume_path_cached = None
    ckpt_saved_opt_groups = None
    try:
        if resume_from_cfg:
            resume_path_cached = resume_from_cfg
        elif auto_resume_latest_cfg:
            resume_path_cached = _find_latest_checkpoint(ckpt_dir)
        if resume_path_cached and os.path.exists(resume_path_cached):
            _tmp = torch.load(resume_path_cached, map_location='cpu', weights_only=False)
            _og = _tmp.get('optimizer_state_dict', {})
            if isinstance(_og, dict):
                ckpt_saved_opt_groups = len(_og.get('param_groups', [])) or 1
            del _tmp
    except Exception:
        pass
    match_groups = os.environ.get('MATCH_OPT_GROUPS_FROM_CKPT', '1') == '1'
    if _get_main_process_flag():
        if ckpt_saved_opt_groups is not None:
            logger.info(f"预读 ckpt 优化器分组数: {ckpt_saved_opt_groups} (MATCH_OPT_GROUPS_FROM_CKPT={int(match_groups)})")
    
    # 为了能够从断点完整恢复 optimizer state：按“主干+回归头”分组；若旧ckpt为单组，_load_checkpoint会重建为单组再恢复
    head_names = ['regression_head', 'param_head', 'head']
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (head_params if any(hn in name for hn in head_names) else backbone_params).append(p)
    head_lr_mult = float(os.environ.get('HEAD_LR_MULT', '4.0'))
    use_single_group = bool(match_groups and (ckpt_saved_opt_groups == 1))
    if use_single_group:
        optimizer = torch.optim.AdamW(module_ref.parameters(), lr=config['lr'])
        if _get_main_process_flag():
            logger.info("优化器分组: 单组(与 ckpt 对齐)")
    else:
        groups = []
        if len(backbone_params) > 0:
            groups.append({'params': backbone_params, 'lr': config['lr']})
        if len(head_params) > 0:
            groups.append({'params': head_params, 'lr': config['lr'] * max(1.0, head_lr_mult)})
        optimizer = torch.optim.AdamW(groups)
        if _get_main_process_flag():
            logger.info(f"优化器分组: 两组(Backbone+Head, HEAD_LR_MULT={head_lr_mult})")
    # IterableDataLoader 可能无 __len__
    try:
        raw_steps_per_epoch = len(train_loader)
    except TypeError:
        raw_steps_per_epoch = int(os.environ.get('ESTIMATED_STEPS_PER_EPOCH', '1000'))
        if _get_main_process_flag():
            logger.warning(f"len(train_loader) 不可用(IterableDataset)。使用估计 steps_per_epoch={raw_steps_per_epoch}，可通过 ESTIMATED_STEPS_PER_EPOCH 覆盖")
    accum_steps = max(1, int(config.get('accum_steps', 1)))
    # 调度器基于“参数更新次数”而非 micro 步数
    steps_per_epoch = max(1, (raw_steps_per_epoch + accum_steps - 1) // accum_steps)
    total_steps = max(1, int(config['epochs']) * steps_per_epoch)
    # LR SCHEDULER: 支持 cosine + warmup（通过环境变量控制）
    sched_name = os.environ.get('LR_SCHEDULER', 'cosine').strip().lower()
    if sched_name in ('cosine_warmup', 'cosine-with-warmup', 'cosinewarmup'):
        warmup_ratio = float(os.environ.get('WARMUP_RATIO', '0.1'))
        warmup_steps_env = int(os.environ.get('WARMUP_STEPS', '0'))
        warmup_steps = warmup_steps_env if warmup_steps_env > 0 else int(total_steps * warmup_ratio)
        warmup_steps = max(1, min(warmup_steps, max(1, total_steps - 1)))
        start_factor = float(os.environ.get('WARMUP_START_FACTOR', '0.01'))
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # AMP 配置
    use_amp = bool(config.get('use_amp', True)) and (device.type == 'cuda')
    amp_dtype_str = str(config.get('amp_dtype', 'bf16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
    rmse_calc = ParameterRMSE()
    # 保存一份CPU版用于数据集标准化，GPU版用于计算
    _param_mean_cpu, _param_std_cpu = param_stats
    param_mean, param_std = _param_mean_cpu.to(device), _param_std_cpu.to(device)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_param_rmse': []}

    # 检查点与续训设置
    ckpt_dir = str(config.get('ckpt_dir', 'finetune_ckpts'))
    ckpt_freq_steps = int(config.get('ckpt_freq_steps', 20000))
    ckpt_freq_epochs = int(config.get('ckpt_freq_epochs', 0))
    resume_from = str(config.get('resume_from', '') or '')
    auto_resume_latest = bool(config.get('auto_resume_latest', True))

    # 步级评估设置
    step_eval_freq = int(config.get('step_eval_freq_steps', 0))
    step_eval_csv = str(config.get('step_eval_csv', os.environ.get('STEP_EVAL_CSV', '/home/share/guofangkeda/wangcunshi/Spectrum/Spec/Spec/pretrain_data/spectrum_tokenized_val_subset.csv')))
    step_eval_workers = int(config.get('step_eval_workers', 0))
    step_eval_rank0_only = bool(config.get('step_eval_rank0_only', True))
    # 子集评估离线缓存到内存（更快）
    step_eval_cache_in_mem = bool(config.get('step_eval_cache_in_mem', True))
    step_eval_max_samples = int(os.environ.get('STEP_EVAL_MAX_SAMPLES', str(config.get('step_eval_max_samples', 1500))))
    # 训练日志频率（按“参数更新步”记数）
    train_log_freq = int(config.get('train_log_every_steps', 0))

    start_epoch = 0
    global_update_steps = 0

    # 恢复
    resume_path = None
    if resume_from:
        resume_path = resume_from
    elif auto_resume_latest:
        resume_path = _find_latest_checkpoint(ckpt_dir)
        if _get_main_process_flag() and resume_path:
            logger.info(f"🔁 自动发现最新检查点: {resume_path}")
    # 使用先前预读到的 resume_path（若有）
    if resume_path_cached:
        resume_path = resume_path_cached
        if _get_main_process_flag():
            logger.info(f"🔁 续训使用预读的检查点路径: {resume_path}")
    if resume_path:
        ckpt = _load_checkpoint(model, optimizer, scheduler, resume_path)
        if ckpt is not None:
            best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            global_update_steps = int(ckpt.get('global_update_steps', 0))
            if _get_main_process_flag():
                logger.info(f"✅ 已从检查点恢复：epoch={start_epoch} global_update_steps={global_update_steps} best_val_loss={best_val_loss:.6f}")
            # 若调度器未成功恢复，则按 gsteps 对齐到正确 LR 阶段
            try:
                resume_info = ckpt.get('_resume_info', {}) if isinstance(ckpt, dict) else {}
                sch_ok = bool(resume_info.get('scheduler_loaded', False))
                opt_ok = bool(resume_info.get('optimizer_loaded', False))
            except Exception:
                sch_ok, opt_ok = False, False
            if (not sch_ok) and (scheduler is not None) and (global_update_steps > 0):
                try:
                    advance = max(0, min(global_update_steps, max(1, total_steps) - 1))
                    for _ in range(advance):
                        scheduler.step()
                    if _get_main_process_flag():
                        logger.warning(f"调度器状态未恢复，已根据 gsteps={global_update_steps} 前推 {advance} 步对齐 LR")
                except Exception:
                    pass
            if (not opt_ok) and _get_main_process_flag():
                logger.warning("优化器状态未恢复，已回退为新优化器（动量未继承，学习率/分组已生效）")
    
    # 若启用步级评估，构建一次性的小验证集 DataLoader（IterableDataset，不使用DistributedSampler）
    step_val_loader = None
    if step_eval_freq > 0:
        try:
            token_to_id_local = load_token_mapping()
            # 仅在rank0构建子集评估数据（可通过开关关闭）
            build_here = True
            if is_distributed and step_eval_rank0_only and (not _get_main_process_flag()):
                build_here = False
            if build_here:
                if step_eval_cache_in_mem:
                    try:
                        if _get_main_process_flag():
                            logger.info("🧲 子集评估采用内存缓存: 首次构建会读取并缓存 %d 个光谱" % step_eval_max_samples)
                        token_to_id_tmp = load_token_mapping()
                        data_small = optimized_preprocess_data(step_eval_csv, token_to_id_tmp, max_samples=step_eval_max_samples, seq_len=config['seq_len'])
                        mem_ds = FinetuneDataset(data_small, _param_mean_cpu.clone(), _param_std_cpu.clone())
                        step_val_loader = DataLoader(
                            mem_ds,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True,
                        )
                    except Exception as e:
                        if _get_main_process_flag():
                            logger.warning(f"内存缓存子集构建失败，回退为流式: {e}")
                        step_eval_cache_in_mem = False
                if not step_eval_cache_in_mem:
                    step_val_dataset = StreamedFinetuneIterableDataset(
                        step_eval_csv,
                        token_to_id_local,
                        config['seq_len'],
                        _param_mean_cpu,
                        _param_std_cpu,
                        is_main_process=_get_main_process_flag(),
                        filter_by_rank=False,
                        world_size=1,
                        global_rank=0,
                    )
                    _kwargs = {
                        'batch_size': config['batch_size'],
                        'shuffle': False,
                        'num_workers': max(0, step_eval_workers),
                        'pin_memory': False,
                        'drop_last': True,
                    }
                    if step_eval_workers > 0:
                        _kwargs['persistent_workers'] = False
                        _kwargs['prefetch_factor'] = 2
                    step_val_loader = DataLoader(step_val_dataset, **_kwargs)
                if _get_main_process_flag():
                    mode_str = 'rank0-only' if (is_distributed and step_eval_rank0_only) else 'all-ranks'
                    cache_str = 'mem-cache' if step_eval_cache_in_mem else 'stream'
                    logger.info(f"🧪 启用步级评估({mode_str},{cache_str}): 每 {step_eval_freq} 步在子集评估 ({step_eval_csv})")
            else:
                step_val_loader = None
        except Exception as e:
            if _get_main_process_flag():
                logger.warning(f"构建步级评估数据集失败，已跳过: {e}")
            step_val_loader = None

    for epoch in range(start_epoch, config['epochs']):
        # 分布式场景设定epoch以确保各卡shuffle一致
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            try:
                train_loader.sampler.set_epoch(epoch)
            except Exception:
                pass
        model.train()
        total_train_loss = 0.0
        total_train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        update_steps = 0
        # 累计当前“参数更新步”内的训练指标（考虑梯度累积）
        train_step_loss_sum = 0.0
        train_step_se_sum = torch.zeros(3, device=device, dtype=torch.float32)
        train_step_count = torch.zeros(1, device=device, dtype=torch.float32)
        
        first_batch_barrier_done = False
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"微调 Epoch {epoch+1}", disable=(not _get_main_process_flag()))):
            seq, params = batch['sequence'].to(device), batch['params'].to(device)
            pos_index = batch.get('pos_index', None)
            if pos_index is not None:
                pos_index = pos_index.to(device)
            # 首批对齐：确保所有rank都已拿到第一批数据后再进入任何DDP通信
            if is_distributed and not first_batch_barrier_done:
                try:
                    dist.barrier()
                except Exception:
                    pass
                first_batch_barrier_done = True
            # 在非最后一个累积步屏蔽同步，降低通信（FSDP/DDP 皆可）
            no_sync_ctx = model.no_sync() if hasattr(model, 'no_sync') and ((batch_idx + 1) % accum_steps != 0) else nullcontext()
            with no_sync_ctx:
                autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                with autocast_ctx:
                    pred_params, raw_loss = model(seq, mode='param_prediction', target_params=params, pos_index=pos_index)
                    loss = raw_loss / accum_steps
                loss.backward()
            # 累计当前“更新步”的训练指标
            train_step_loss_sum += float(raw_loss.item())
            pred_unnorm_t = pred_params * param_std + param_mean
            true_unnorm_t = params * param_std + param_mean
            se_t = (pred_unnorm_t - true_unnorm_t) ** 2
            train_step_se_sum += se_t.sum(dim=0)
            train_step_count += torch.as_tensor([seq.size(0)], device=device, dtype=torch.float32)
            total_train_loss += float(loss.item()) * accum_steps
            total_train_batches += 1

            # 到达累积步数，进行一次参数更新与LR步进
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_steps += 1
                global_update_steps += 1
                # 训练步级日志
                if train_log_freq > 0 and (global_update_steps % train_log_freq == 0):
                    # 跨卡聚合
                    if is_distributed:
                        sl = torch.as_tensor([train_step_loss_sum, float(accum_steps)], device=device, dtype=torch.float32)
                        dist.all_reduce(sl, op=dist.ReduceOp.SUM)
                        step_avg_train_loss = (sl[0] / torch.clamp(sl[1], min=1.0)).item()
                        dist.all_reduce(train_step_se_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(train_step_count, op=dist.ReduceOp.SUM)
                    else:
                        step_avg_train_loss = (train_step_loss_sum / max(1, accum_steps))
                    if train_step_count.item() > 0:
                        step_train_rmse = torch.sqrt(train_step_se_sum / train_step_count.item()).tolist()
                        train_rmse_dict = {
                            'Teff': step_train_rmse[0],
                            'logg': step_train_rmse[1],
                            'Fe/H': step_train_rmse[2],
                            'total': float(np.sqrt(np.mean([v**2 for v in step_train_rmse])))
                        }
                    else:
                        train_rmse_dict = {'Teff': 0.0, 'logg': 0.0, 'Fe/H': 0.0, 'total': 0.0}
                    if _get_main_process_flag():
                        extra = ''
                        if os.environ.get('PRINT_NORM_RMSE', '0') == '1' and 'norm_Teff' in train_rmse_dict:
                            extra = f" | norm_RMSE: Teff={train_rmse_dict['norm_Teff']:.6f}, logg={train_rmse_dict['norm_logg']:.6f}, Fe/H={train_rmse_dict['norm_Fe/H']:.6f}"
                        logger.info(f"🟢 Step {global_update_steps} | 训练: 损失 {step_avg_train_loss:.6f} | RMSE: Teff={train_rmse_dict['Teff']:.6f}, logg={train_rmse_dict['logg']:.6f}, Fe/H={train_rmse_dict['Fe/H']:.6f}, total={train_rmse_dict['total']:.6f}{extra}")
                        try:
                            logger.info(json.dumps({'event':'train_step','step':int(global_update_steps),'loss':float(step_avg_train_loss),'rmse':train_rmse_dict}, ensure_ascii=False))
                        except Exception:
                            pass
                    # 重置累计量以便下一个更新步
                    train_step_loss_sum = 0.0
                    train_step_se_sum.zero_()
                    train_step_count.zero_()
                # 按步保存（仅主进程）
                if ckpt_freq_steps > 0 and (global_update_steps % ckpt_freq_steps == 0):
                    _save_checkpoint(model, optimizer, scheduler, epoch, global_update_steps, best_val_loss, config, amp_dtype, ckpt_dir, tag=f"step{global_update_steps:08d}")
                # 步级评估（全卡聚合）
                if step_val_loader is not None and step_eval_freq > 0 and (global_update_steps % step_eval_freq == 0):
                    model.eval()
                    step_total_loss = 0.0
                    step_total_batches = 0
                    se_sum_step = torch.zeros(3, device=device, dtype=torch.float32)
                    se_count_step = torch.zeros(1, device=device, dtype=torch.float32)
                    with torch.no_grad():
                        for vbatch in step_val_loader:
                            vseq, vparams = vbatch['sequence'].to(device), vbatch['params'].to(device)
                            vpos = vbatch.get('pos_index', None)
                            if vpos is not None:
                                vpos = vpos.to(device)
                            autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                            with autocast_ctx:
                                vpred, vloss = model(vseq, mode='param_prediction', target_params=vparams, pos_index=vpos)
                            step_total_loss += float(vloss.item())
                            step_total_batches += 1
                            vpred_unnorm = vpred * param_std + param_mean
                            vtrue_unnorm = vparams * param_std + param_mean
                            vse = (vpred_unnorm - vtrue_unnorm) ** 2
                            se_sum_step += vse.sum(dim=0)
                            se_count_step += torch.as_tensor([vseq.size(0)], device=device, dtype=torch.float32)
                    if is_distributed and (not step_eval_rank0_only):
                        sl = torch.as_tensor([step_total_loss, float(step_total_batches)], device=device, dtype=torch.float32)
                        dist.all_reduce(sl, op=dist.ReduceOp.SUM)
                        step_avg_loss = (sl[0] / torch.clamp(sl[1], min=1.0)).item()
                        dist.all_reduce(se_sum_step, op=dist.ReduceOp.SUM)
                        dist.all_reduce(se_count_step, op=dist.ReduceOp.SUM)
                    else:
                        step_avg_loss = (step_total_loss / max(1, step_total_batches))
                    if se_count_step.item() > 0:
                        rmse_each = torch.sqrt(se_sum_step / se_count_step.item()).tolist()
                        step_rmse = {
                            'Teff': rmse_each[0],
                            'logg': rmse_each[1],
                            'Fe/H': rmse_each[2],
                            'total': float(np.sqrt(np.mean([v**2 for v in rmse_each])))
                        }
                    else:
                        step_rmse = {'Teff': 0.0, 'logg': 0.0, 'Fe/H': 0.0, 'total': 0.0}
                    # 可选打印“标准化空间 RMSE”（用于与旧口径对齐对比）
                    if os.environ.get('PRINT_NORM_RMSE', '0') == '1' and se_count_step.item() > 0:
                        norm_rmse_vec = torch.sqrt((se_sum_step / se_count_step.item()) / (param_std ** 2 + 1e-12)).tolist()
                        step_rmse.update({'norm_Teff': norm_rmse_vec[0], 'norm_logg': norm_rmse_vec[1], 'norm_Fe/H': norm_rmse_vec[2]})
                    if _get_main_process_flag():
                        extra = ''
                        if os.environ.get('PRINT_NORM_RMSE', '0') == '1' and 'norm_Teff' in step_rmse:
                            extra = f" | norm_RMSE: Teff={step_rmse['norm_Teff']:.6f}, logg={step_rmse['norm_logg']:.6f}, Fe/H={step_rmse['norm_Fe/H']:.6f}"
                        logger.info(f"📊 Step {global_update_steps} | 子集验证: 损失 {step_avg_loss:.6f} | RMSE: Teff={step_rmse['Teff']:.6f}, logg={step_rmse['logg']:.6f}, Fe/H={step_rmse['Fe/H']:.6f}, total={step_rmse['total']:.6f}{extra}")
                        try:
                            logger.info(json.dumps({'event':'step_eval','step':int(global_update_steps),'loss':float(step_avg_loss),'rmse':step_rmse}, ensure_ascii=False))
                        except Exception:
                            pass
                    # 基于子集验证loss的 Top-3 best（仅主进程）
                    if np.isfinite(step_avg_loss):
                        _update_step_topk_best(model, step_avg_loss, global_update_steps, ckpt_dir, topk=3)
                        if _get_main_process_flag():
                            logger.info("  ✅ 子集验证Top-K候选已更新 (基于avg_val_loss)")
                        # 同时更新全局best（与epoch级别共用阈值）
                        if step_avg_loss < best_val_loss and _get_main_process_flag():
                            best_val_loss = step_avg_loss
                            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                            torch.save(to_save, 'best_finetuned_model_v2.pth')
                            logger.info("  ✅ 保存最佳模型（基于子集验证）")
                    model.train()

        model.eval()
        total_val_loss = 0.0
        total_val_batches = 0
        # 逐卡累计未归一化参数的平方误差和与样本计数，用于跨卡聚合
        se_sum = torch.zeros(3, device=device, dtype=torch.float32)
        se_count = torch.zeros(1, device=device, dtype=torch.float32)
        with torch.no_grad():
            for batch in val_loader:
                seq, params = batch['sequence'].to(device), batch['params'].to(device)
                vpos = batch.get('pos_index', None)
                if vpos is not None:
                    vpos = vpos.to(device)
                autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
                with autocast_ctx:
                    pred_params, loss = model(seq, mode='param_prediction', target_params=params, pos_index=vpos)
                total_val_loss += float(loss.item())
                total_val_batches += 1
                # 反标准化后累计平方误差
                pred_unnorm = pred_params * param_std + param_mean
                true_unnorm = params * param_std + param_mean
                sq_err = (pred_unnorm - true_unnorm) ** 2
                se_sum += sq_err.sum(dim=0)
                se_count += torch.as_tensor([seq.size(0)], device=device, dtype=torch.float32)

        # 计算平均指标（跨卡聚合）
        if is_distributed:
            # 训练损失
            tl = torch.as_tensor([total_train_loss, float(total_train_batches)], device=device, dtype=torch.float32)
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
            avg_train_loss = (tl[0] / torch.clamp(tl[1], min=1.0)).item()
            # 验证损失
            vl = torch.as_tensor([total_val_loss, float(total_val_batches)], device=device, dtype=torch.float32)
            dist.all_reduce(vl, op=dist.ReduceOp.SUM)
            avg_val_loss = (vl[0] / torch.clamp(vl[1], min=1.0)).item()
            # RMSE聚合
            dist.all_reduce(se_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(se_count, op=dist.ReduceOp.SUM)
        else:
            avg_train_loss = (total_train_loss / max(1, total_train_batches))
            avg_val_loss = (total_val_loss / max(1, total_val_batches))

        # 计算跨卡RMSE
        if se_count.item() > 0:
            rmse_each = torch.sqrt(se_sum / se_count.item()).tolist()
            avg_val_rmse = {
                'Teff': rmse_each[0],
                'logg': rmse_each[1],
                'Fe/H': rmse_each[2],
                'total': float(np.sqrt(np.mean([v**2 for v in rmse_each])))
            }
        else:
            avg_val_rmse = {'Teff': 0.0, 'logg': 0.0, 'Fe/H': 0.0, 'total': 0.0}
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_param_rmse'].append(avg_val_rmse)
        
        if _get_main_process_flag():
            logger.info(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证RMSE: {avg_val_rmse['total']:.4f}")

        if avg_val_loss < best_val_loss and _get_main_process_flag():
            best_val_loss = avg_val_loss
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(to_save, 'best_finetuned_model_v2.pth')
            logger.info("  ✅ 保存最佳模型")
        # 按轮保存（仅主进程）
        if ckpt_freq_epochs > 0 and ((epoch + 1) % ckpt_freq_epochs == 0):
            _save_checkpoint(model, optimizer, scheduler, epoch + 1, global_update_steps, best_val_loss, config, amp_dtype, ckpt_dir, tag=f"epoch{epoch+1:04d}")
            
    return history

def main():
    """主函数"""
    print("🚀 优化微调脚本 (V2) - 修正参数解析 | 支持多卡DDP")
    
    config = {
        'batch_size': int(os.environ.get('BATCH_SIZE', '16')),
        'epochs': int(os.environ.get('EPOCHS', '1')),
        'lr': float(os.environ.get('LR', '5e-5')),
        'freeze_layers': int(os.environ.get('FREEZE_LAYERS', '4')),
        'seq_len': int(os.environ.get('SEQ_LEN', '8192')),
        'max_train_samples': None,
        'max_val_samples': None,
        'num_workers': int(os.environ.get('NUM_WORKERS', '1')),
        # 新增：从环境变量读取梯度累积与AMP设置
        'accum_steps': int(os.environ.get('ACCUM_STEPS', '1')),
        'use_amp': os.environ.get('USE_AMP', '1') == '1',
        'amp_dtype': os.environ.get('AMP_DTYPE', 'bf16'),
        # 检查点配置
        'ckpt_dir': os.environ.get('CKPT_DIR', 'finetune_ckpts'),
        'ckpt_freq_steps': int(os.environ.get('CKPT_FREQ_STEPS', '0')),
        'ckpt_freq_epochs': int(os.environ.get('CKPT_FREQ_EPOCHS', '1')),
        'resume_from': os.environ.get('RESUME_FROM', ''),
        'auto_resume_latest': os.environ.get('AUTO_RESUME_LATEST', '1') == '1',
        'ckpt_compat_pretrain_naming': os.environ.get('CKPT_COMPAT_PRETRAIN_NAMING', '1') == '1',
        # 步级评估配置
        'step_eval_freq_steps': int(os.environ.get('STEP_EVAL_FREQ_STEPS', '500')),
        'step_eval_csv': os.environ.get('STEP_EVAL_CSV', '/home/share/guofangkeda/wangcunshi/Spectrum/Spec/Spec/pretrain_data/spectrum_tokenized_val_subset.csv'),
        'step_eval_workers': int(os.environ.get('STEP_EVAL_WORKERS', '0')),
        # 新增：训练步级日志频率（单位：参数更新步）；0 表示关闭
        'train_log_every_steps': int(os.environ.get('TRAIN_LOG_EVERY_STEPS', '20')),
    }

    # 分布式初始化
    world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size_env > 1
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    global_rank = int(os.environ.get('RANK', '0'))
    if is_distributed:
        torch.cuda.set_device(local_rank)
        if not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size_env, rank=global_rank)

    # 分布式时不再强制 batch_size=1，沿用环境变量/默认值

    # 控制日志输出级别（仅主进程信息级，其他进程降级为WARNING）
    if not _get_main_process_flag():
        logger.setLevel(logging.WARNING)
        for h in list(logger.handlers):
            try:
                h.setLevel(logging.WARNING)
            except Exception:
                pass
        # 非主进程移除文件写入，避免多进程竞争一个日志文件导致阻塞
        for h in list(logger.handlers):
            try:
                import logging as _logging
                if isinstance(h, _logging.FileHandler):
                    logger.removeHandler(h)
            except Exception:
                pass

    # 加载模型和数据
    base_model = load_pretrained_model(config['seq_len'])
    # 在DDP包裹前，先触发一次参数回归头的创建，避免训练首步才动态注册参数导致DDP未跟踪
    try:
        with torch.no_grad():
            _warm_seq = torch.zeros((1, max(2, min(16, config['seq_len']))), dtype=torch.long)
            _warm_params = torch.zeros((1, 3), dtype=torch.float32)
            base_model(_warm_seq, mode='param_prediction', target_params=_warm_params)
    except Exception:
        pass
    if is_distributed:
        model = DDP(base_model.to(torch.device(f"cuda:{local_rank}")), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        try:
            model._set_static_graph()
        except Exception:
            pass
    else:
        model = base_model
    token_to_id = load_token_mapping()

    train_csv = os.environ.get('TRAIN_CSV', '/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune_data/spectrum_tokenized_train.csv')
    val_csv = os.environ.get('VAL_CSV', '/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune_data/spectrum_tokenized_val.csv')

    # 优先使用离线分片（每个 rank 只读自己的 shard），否则回退到原始 CSV
    train_csv_resolved, train_using_shards = _resolve_sharded_path(train_csv, role='train', world_size=world_size_env, global_rank=global_rank)
    val_csv_resolved, val_using_shards = _resolve_sharded_path(val_csv, role='val', world_size=world_size_env, global_rank=global_rank)
    if _get_main_process_flag():
        if train_using_shards or val_using_shards:
            logger.info(f"📦 使用分片CSV: train={train_csv_resolved}, val={val_csv_resolved}")
        else:
            logger.warning("未检测到分片CSV，回退到原始CSV（注意将导致多卡重复扫描I/O）")

    # 流式估计参数统计（前 N 个光谱）
    stats_samples = int(os.environ.get('PARAM_STATS_SAMPLES', '2000'))
    if _get_main_process_flag():
        logger.info(f"📏 流式统计参数: 取前 {stats_samples} 个光谱估计 mean/std（可被环境变量覆盖）")
    # 允许从环境变量覆盖mean/std，以便A/B两段使用一致的标准化
    override = _load_param_stats_override_from_env()
    if override is not None:
        param_mean, param_std = override
        if _get_main_process_flag():
            logger.info(f"🔧 使用覆盖的参数统计: mean={param_mean.tolist()} std={param_std.tolist()}")
    else:
        param_mean, param_std = compute_param_stats_streaming(train_csv_resolved, token_to_id, config['seq_len'], max_samples_for_stats=stats_samples)

    # 优先：内存预加载（小数据量，稳定且不依赖chunk）
    preload_to_memory = os.environ.get('PRELOAD_TO_MEMORY', '1') == '1'
    use_online_dispatch = False
    if preload_to_memory:
        if _get_main_process_flag():
            logger.info("🧲 内存预加载训练/验证集（不分chunk）")
        data_mem_train = optimized_preprocess_data(
            train_csv_resolved, token_to_id, max_samples=None, seq_len=config['seq_len']
        )
        train_dataset = FinetuneDataset(data_mem_train, param_mean.clone(), param_std.clone())
        try:
            data_val_mem = optimized_preprocess_data(
                val_csv_resolved, token_to_id, max_samples=None, seq_len=config['seq_len']
            )
        except Exception:
            data_val_mem = []
        val_dataset = FinetuneDataset(data_val_mem, param_mean.clone(), param_std.clone()) if len(data_val_mem) > 0 else None
    else:
        # 根据 ONLINE_DISPATCH 选择在线分发或本地流式
        use_online_dispatch = (os.environ.get('ONLINE_DISPATCH', '0') == '1') and is_distributed
        if use_online_dispatch:
            if _get_main_process_flag():
                logger.info("🔌 启用在线分发模式：rank0 统一读取并轮转派发到各 rank")
            train_dataset = OnlineDispatchIterableDataset(
                train_csv_resolved, token_to_id, config['seq_len'], param_mean, param_std,
                world_size=world_size_env, global_rank=global_rank
            )
            val_dataset = OnlineDispatchIterableDataset(
                val_csv_resolved, token_to_id, config['seq_len'], param_mean, param_std,
                world_size=world_size_env, global_rank=global_rank
            )
        else:
            # 构建流式 IterableDataset（按rank/worker过滤，避免重复IO）
            train_dataset = StreamedFinetuneIterableDataset(
                train_csv_resolved, token_to_id, config['seq_len'], param_mean, param_std,
                is_main_process=_get_main_process_flag(),
                filter_by_rank=(is_distributed and (not train_using_shards)), world_size=world_size_env, global_rank=global_rank
            )
            val_dataset = StreamedFinetuneIterableDataset(
                val_csv_resolved, token_to_id, config['seq_len'], param_mean, param_std,
                is_main_process=_get_main_process_flag(),
                filter_by_rank=(is_distributed and (not val_using_shards)), world_size=world_size_env, global_rank=global_rank
            )

    # DataLoader（IterableDataset 不使用 DistributedSampler）
    # 读取 STREAM_* 环境变量以覆盖 DataLoader 行为
    _sw = os.environ.get('STREAM_WORKERS', '').strip()
    # 参考预训练脚本：当 ONLINE_DISPATCH=0 时，即使是分布式也允许开启 DataLoader 多 worker 以提高并行度
    # 仅在 ONLINE_DISPATCH=1 时强制 0 worker，避免与点对点分发冲突
    stream_workers = 0 if use_online_dispatch else (int(_sw) if _sw.isdigit() else 0)
    _spf = os.environ.get('STREAM_PREFETCH', '').strip()
    stream_prefetch = int(_spf) if _spf.isdigit() else None
    stream_pin_memory = os.environ.get('STREAM_PIN_MEMORY', '0') == '1'
    stream_persistent = os.environ.get('STREAM_PERSISTENT', '0') == '1'
    streaming_mode_flag = os.environ.get('STREAMING_MODE', '1') == '1'
    if not streaming_mode_flag and _get_main_process_flag():
        logger.warning("STREAMING_MODE=0：当前版本仍使用流式 IterableDataset（功能开关占位，不切换为预加载模式）。")

    train_num_workers = max(0, int(stream_workers))
    val_num_workers = max(0, train_num_workers // 2)
    train_persistent = bool(stream_persistent and (train_num_workers > 0))
    val_persistent = bool(stream_persistent and (val_num_workers > 0))

    common_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'pin_memory': stream_pin_memory,
        'drop_last': True,
    }
    # 训练 DataLoader
    train_kwargs = dict(common_kwargs)
    # 内存数据集可打乱；IterableDataset 不允许
    train_kwargs['shuffle'] = False if isinstance(train_dataset, torch.utils.data.IterableDataset) else True
    train_kwargs['num_workers'] = train_num_workers
    train_kwargs['persistent_workers'] = train_persistent
    if (stream_prefetch is not None) and (train_num_workers > 0):
        train_kwargs['prefetch_factor'] = stream_prefetch
    train_loader = DataLoader(train_dataset, **train_kwargs)

    # 验证 DataLoader（禁用 shuffle）
    val_kwargs = dict(common_kwargs)
    val_kwargs['shuffle'] = False
    val_kwargs['num_workers'] = val_num_workers
    val_kwargs['persistent_workers'] = val_persistent
    if (stream_prefetch is not None) and (val_num_workers > 0):
        val_kwargs['prefetch_factor'] = stream_prefetch
    val_loader = DataLoader(val_dataset, **val_kwargs)
    
    # 开始微调
    history = finetune(model, train_loader, val_loader, config, (param_mean, param_std))
    
    # 保存结果（仅主进程）
    if _get_main_process_flag():
        with open('finetune_history_v2.json', 'w') as f:
            json.dump(history, f, indent=2)
        logger.info("🎉 优化微调完成！")

    # 收尾
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()