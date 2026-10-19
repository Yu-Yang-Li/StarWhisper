"""
光谱扩散模型 - 基于AO-GPT-MDM论文的真正扩散实现
参考：Any-Order GPT as Masked Diffusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, Tuple

def modulate(x, shift, scale):
    """AdaLN调制函数"""
    return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    """RMS归一化 - 兼容旧版PyTorch"""
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        # 兼容性实现：旧版PyTorch没有F.rms_norm
        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
        else:
            # 手动实现RMSNorm
            norm = x.norm(dtype=torch.float32, dim=-1, keepdim=True)
            rms = norm / (x.size(-1) ** 0.5)
            return (x / (rms + self.eps)) * self.weight

class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力（带QK归一化）"""
    def __init__(self, n_embd, n_head, block_size, use_qk_norm=True):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # QKV投影
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # QK归一化
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        
        # Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(block_size + 1, block_size + 1))
                                .view(1, 1, block_size + 1, block_size + 1))

    def forward(self, x):
        B, T, C = x.size()
        
        # QKV计算
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # QK归一化
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 注意力计算
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """MLP模块"""
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DiffusionTransformerBlock(nn.Module):
    """扩散Transformer块（带AdaLN条件注入）"""
    def __init__(self, n_embd, n_head, block_size, cond_dim=128):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = OptimizedMultiHeadAttention(n_embd, n_head, block_size, use_qk_norm=True)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)
        
        # AdaLN条件投影 - 6个参数：scale, shift, gate for attn and mlp
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * n_embd, bias=True)
        )

    def forward(self, x, cond):
        """
        Args:
            x: [B, T, n_embd] 输入特征
            cond: [B, T, cond_dim] 目标位置条件
        """
        # AdaLN参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(cond).chunk(6, dim=-1)
        
        # 注意力块（带AdaLN调制）
        x = x + gate_msa * self.attn(modulate(self.ln1(x), shift_msa, scale_msa))
        
        # MLP块（带AdaLN调制）
        x = x + gate_mlp * self.mlp(modulate(self.ln2(x), shift_mlp, scale_mlp))
        
        return x

class FinalLayer(nn.Module):
    """最终层（带AdaLN）"""
    def __init__(self, n_embd, cond_dim=128):
        super().__init__()
        self.ln_f = RMSNorm(n_embd)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * n_embd, bias=True)
        )
    
    def forward(self, x, cond):
        shift, scale = self.adaLN_final(cond).chunk(2, dim=-1)
        x = modulate(self.ln_f(x), shift, scale)
        return x

class SpectrumDiffusionModel(nn.Module):
    """
    光谱扩散模型 - 真正的AO-GPT风格实现
    
    核心特性：
    1. 目标位置感知条件（wtpe）
    2. Any-Order生成能力
    3. AdaLN条件注入
    4. [None] token机制
    """
    
    def __init__(self, vocab_size=84, n_embd=256, n_head=8, n_layer=6, 
                 block_size=12288, cond_dim=128,
                 bos_token_id: int = 0, eos_token_id: int = 1, pad_token_id: int = 2,
                 mask_token_id: int | None = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.cond_dim = cond_dim
        # 常用特殊token id（由外部注入，默认与词表约定一致）
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id if mask_token_id is not None else pad_token_id
        
        # 嵌入层
        self.wte = nn.Embedding(vocab_size, n_embd)  # token嵌入
        self.wpe = nn.Embedding(block_size + 1, n_embd)  # 位置嵌入（+1为[None] token）
        self.wtpe = nn.Embedding(block_size, cond_dim)  # 目标位置嵌入（扩散条件）
        self.wnonee = nn.Embedding(1, n_embd)  # [None] token嵌入
        
        # Transformer块
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(n_embd, n_head, block_size, cond_dim)
            for _ in range(n_layer)
        ])
        
        # 最终层
        self.final_layer = FinalLayer(n_embd, cond_dim)
        
        # 语言模型头
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)
        
        # 回归头（可选）
        self.regression_head = None
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"✅ 光谱扩散模型创建完成")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   嵌入维度: {n_embd}")
        print(f"   注意力头数: {n_head}")
        print(f"   层数: {n_layer}")
        print(f"   最大序列长度: {block_size}")
        print(f"   条件维度: {cond_dim}")
        print(f"   参数数量: {self.get_num_params():,}")

    def _init_weights(self, module):
        """初始化权重（使用截断正态分布）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, 
                                       a=-3*0.02, b=3*0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, 
                                       a=-3*0.02, b=3*0.02)

    def sample_random_orders(self, x):
        """采样结构化随机顺序（仅在有效区[1..EOS-1]内打乱；PAD保留在末尾）"""
        batch_size, seq_length = x.shape[0], x.shape[1]
        shuffled_orders: list[torch.Tensor] = []
        for b in range(batch_size):
            order = self._create_structured_random_order_from_sample(x[b])
            shuffled_orders.append(order)
        return torch.stack(shuffled_orders).to(x.device)
    
    def _create_structured_random_order_from_sample(self, tokens: torch.Tensor) -> torch.Tensor:
        """基于样本内容创建结构化随机顺序：
        - BOS 固定在开头
        - 仅在有效区 [1..EOS-1] 内按4位组打乱（保持组内顺序）
        - PAD（EOS 之后）固定保留在序列末尾
        """
        device = tokens.device
        t = int(tokens.size(0))
        if t <= 2:
            return torch.arange(t, device=device)

        bos_id = getattr(self, 'bos_token_id', 0)
        eos_id = getattr(self, 'eos_token_id', 1)

        # 寻找该样本的 EOS 位置；若不存在，退回最后位置
        eos_pos_tensor = (tokens == eos_id).nonzero(as_tuple=False)
        if eos_pos_tensor.numel() > 0:
            eos_pos = int(eos_pos_tensor[0, 0].item())
        else:
            eos_pos = t - 1

        # 有效区间 [1 .. eos_pos-1]
        start_idx = 1
        end_idx = max(start_idx, eos_pos - 1)
        valid_len = max(0, end_idx - start_idx + 1)
        if valid_len <= 0:
            # 仅有 BOS/EOS，无需打乱；其余认为是 PAD
            order = list(range(0, min(t, eos_pos + 1)))
            # 追加PAD区
            for i in range(eos_pos + 1, t):
                order.append(i)
            return torch.tensor(order, device=device)

        num_flux_groups = valid_len // 4
        order: list[int] = [0]  # BOS在开头
        if num_flux_groups > 0:
            flux_group_order = torch.randperm(num_flux_groups, device=device)
            for group_idx in flux_group_order:
                base = start_idx + int(group_idx.item()) * 4
                for k in range(4):
                    pos = base + k
                    if pos <= end_idx:
                        order.append(pos)

        # 处理不足4的尾部
        remaining_start = start_idx + num_flux_groups * 4
        for i in range(remaining_start, end_idx + 1):
            order.append(i)

        # 追加 EOS
        if 0 <= eos_pos < t:
            order.append(eos_pos)

        # 追加 PAD 段（EOS 之后）
        for i in range(eos_pos + 1, t):
            order.append(i)

        return torch.tensor(order, device=device)

    def sample_random_orders_CL(self, x, random_ratio):
        """带课程学习的结构化随机顺序采样"""
        batch_size, seq_length = x.shape[0], x.shape[1]
        shuffled_orders: list[torch.Tensor] = []
        for b in range(batch_size):
            if random.random() < random_ratio:
                order = self._create_structured_random_order_from_sample(x[b])
            else:
                order = torch.arange(seq_length, device=x.device)
            shuffled_orders.append(order)
        return torch.stack(shuffled_orders).to(x.device)

    def set_ascending_orders(self, x):
        """设置升序（标准自回归）"""
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []

        for _ in range(batch_size):
            shuffled_orders.append(torch.arange(seq_length, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)
    
    def shuffle(self, x, orders):
        """根据顺序打乱张量"""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x
    
    def unshuffle(self, shuffled_x, orders):
        """根据原始顺序恢复张量"""
        batch_size, seq_len = shuffled_x.shape[:2]
        device = shuffled_x.device
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def forward(self, idx, mode='Random_CL', orders=None, random_ratio=0.9, target_params=None, loss_fn=None, abs_pos_seq=None, pos_index=None):
        """
        前向传播 - 真正的扩散模型实现
        
        Args:
            idx: [B, T] 输入token序列
            mode: 训练模式 - 'AR', 'Random', 'Random_CL', 'param_prediction'
            orders: 自定义顺序
            random_ratio: Random_CL模式的随机比例
            target_params: [B, num_params] 目标参数（用于微调）
            loss_fn: 自定义损失函数
        
        Returns:
            logits / pred_params, loss
        """
        if mode == 'param_prediction':
            return self.forward_regression(idx, target_params)

        if mode is None:
            assert orders is not None and idx.shape == orders.shape
        elif mode == 'AR':
            orders = self.set_ascending_orders(idx)
        elif mode == 'Random':
            orders = self.sample_random_orders(idx)
        elif mode == 'Random_CL':
            assert random_ratio is not None
            orders = self.sample_random_orders_CL(idx, random_ratio)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 若显式提供了 pos_index，优先使用；兼容旧入参名 abs_pos_seq（视作pos_index）
        if pos_index is None and abs_pos_seq is not None:
            pos_index = abs_pos_seq
        return self.forward_fn(idx, orders, loss_fn, pos_index=pos_index)

    def forward_fn(self, idx, orders, loss_fn=None, pos_index=None):
        """
        扩散模型的核心前向传播
        基于AO-GPT-MDM论文的实现
        
        Args:
            idx: 输入token序列
            orders: 顺序
            loss_fn: 自定义损失函数，如果为None则使用交叉熵
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"序列长度 {t} 超过最大长度 {self.block_size}"
        
        # 位置序列（包含[None] token）- 仅在未提供pos_index时使用相对位次
        pos = torch.arange(0, t + 1, dtype=torch.long, device=device)
        
        # 1. 打乱输入序列
        idx_shuffled = self.shuffle(idx, orders)  # [b, t]
        targets = idx_shuffled  # 目标就是打乱后的序列
        
        # 2. 准备token嵌入
        tok_emb = self.wte(idx_shuffled)  # [b, t, n_embd]
        
        # 3. 添加[None] token嵌入
        none_tok_emb = self.wnonee(torch.tensor([[0]], device=device))  # [1, 1, n_embd]
        none_tok_emb = none_tok_emb.expand(b, -1, -1)  # [b, 1, n_embd]
        tok_emb = torch.cat([none_tok_emb, tok_emb], dim=1)  # [b, t+1, n_embd]
        
        # 4/5/6. 位置与条件嵌入
        if pos_index is not None:
            # pos_index: [B,T] in [0, block_size-1]
            pos_index = pos_index.long().clamp(min=0, max=self.block_size - 1)
            pos_shuf = self.shuffle(pos_index, orders)  # [B,T]

            # 位置嵌入（与token拼到x上）
            wpe_tok = self.wpe(pos_shuf)  # [B,T,n_embd]
            # [None]位置：每条样本的最大索引
            max_idx = torch.clamp(pos_index.max(dim=1, keepdim=True).values, max=self.block_size - 1)  # [B,1]
            none_pos_emb = self.wpe(max_idx.squeeze(1)).unsqueeze(1)  # [B,1,n_embd]
            pos_full = torch.cat([none_pos_emb, wpe_tok], dim=1)  # [B,T+1,n_embd]
            x = tok_emb + pos_full

            # 条件嵌入（wtpe）
            wtpe_tok = self.wtpe(pos_shuf)  # [B,T,cond_dim]
            none_cond = self.wtpe(max_idx.squeeze(1)).unsqueeze(1)  # [B,1,cond_dim]
            target_pos_emb_final = torch.cat([wtpe_tok, none_cond], dim=1)  # [B,T+1,cond_dim]
        else:
            # 旧逻辑：相对位次
            pos_emb = self.wpe(pos)  # [t+1, n_embd]
            pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
            pos_emb_prefix = pos_emb[:, :1]
            pos_emb_postfix = self.shuffle(pos_emb[:, 1:], orders)
            x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

            target_pos_emb = self.wtpe(pos[:t]).unsqueeze(0).expand(b, -1, -1)
            target_pos_emb_shuffled = self.shuffle(target_pos_emb, orders)
            none_cond = self.wtpe(torch.tensor([min(t, self.block_size-1)], device=device)).unsqueeze(0).expand(b, -1, -1)
            target_pos_emb_final = torch.cat([target_pos_emb_shuffled, none_cond], dim=1)
        
        # 7. Transformer块（带条件）
        for block in self.blocks:
            x = block(x, target_pos_emb_final)
        
        # 8. 最终层
        x = self.final_layer(x, target_pos_emb_final)
        
        # 9. 语言模型头
        logits = self.lm_head(x)  # [b, t+1, vocab_size]
        
        # 10. 计算损失（移除[None] token的预测）
        shift_logits = logits[..., :-1, :].contiguous()  # [b, t, vocab_size]
        
        if loss_fn is not None:
            # 使用自定义损失函数
            loss = loss_fn(shift_logits, targets)
        else:
            # 使用默认交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss

    def add_regression_head(self, num_params=3):
        """添加参数回归头"""
        self.regression_head = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_embd, num_params)
        )
        
        device = next(self.parameters()).device
        self.regression_head = self.regression_head.to(device)
        print(f"✅ 添加参数回归头 (参数数: {num_params})")

    def forward_regression(self, idx, target_params=None, abs_pos_seq: torch.Tensor | None = None):
        """回归任务的前向传播"""
        if self.regression_head is None:
            self.add_regression_head()

        B, T = idx.size()
        assert T <= self.block_size, f"序列长度 {T} 超过最大长度 {self.block_size}"
        
        # 标准前向传播（支持绝对位置条件）
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # 嵌入
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos[:, :T])  # 确保pos长度匹配
        x = tok_emb + pos_emb
        
        # 条件：若提供 abs_pos_seq，则用 wtpe(abs_pos_seq) 作为条件；否则用零条件
        if abs_pos_seq is not None:
            # 期望形状 [B, T]
            abs_pos_seq = abs_pos_seq.clamp_(min=0, max=self.block_size-1)
            cond = self.wtpe(abs_pos_seq)
        else:
            cond = torch.zeros(B, T, self.cond_dim, device=idx.device)
        
        for block in self.blocks:
            x = block(x, cond)
        
        # 最终层
        x = self.final_layer(x, cond)
        
        # 全局平均池化
        features = x.mean(dim=1)
        
        # 回归头
        pred_params = self.regression_head(features)
        
        loss = None
        if target_params is not None:
            loss = F.mse_loss(pred_params, target_params)
            
        return pred_params, loss

    def generate_parallel(self, initial_tokens, num_steps=64, temperature=1.0, top_p=1.0,
                          gen_mask: torch.Tensor | None = None, generate_from: int | None = None):
        """
        并行生成（扩散模型的推理）
        基于论文中的迭代去噪过程
        """
        device = initial_tokens.device
        b, t = initial_tokens.shape
        
        # 初始化：将生成区域显式置为[MASK]，其余保留原值
        # 使用模型注入的 mask/pad id，避免把<BOS>(0)误当作MASK
        current_tokens = initial_tokens.clone()
        mask_token_id = getattr(self, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = getattr(self, 'pad_token_id', 2)

        # 构造初始生成区域：优先使用外部 gen_mask，其次使用 generate_from，再次使用 PAD 区域
        if gen_mask is None:
            if generate_from is not None:
                start = int(generate_from)
                gen_mask = torch.zeros_like(initial_tokens, dtype=torch.bool, device=device)
                if start < t:
                    gen_mask[:, start:] = True
            else:
                pad_id = getattr(self, 'pad_token_id', 2)
                gen_mask = (initial_tokens == pad_id)
        else:
            # 保证设备/类型一致
            gen_mask = gen_mask.to(device=device, dtype=torch.bool)

        # 绝不覆盖特殊符号
        gen_mask = gen_mask & (initial_tokens != self.bos_token_id) & (initial_tokens != self.eos_token_id)
        # 先把目标区域置为MASK，确保后续“去掩码”流程生效
        if gen_mask.any():
            current_tokens = current_tokens.clone()
            current_tokens[gen_mask] = mask_token_id
        
        for step in range(num_steps):
            # 计算当前时间步的mask比例
            mask_ratio = 1.0 - (step / num_steps)
            
            # 前向传播获取预测
            with torch.no_grad():
                logits, _ = self.forward(current_tokens, mode='AR')
                logits = logits[:, :-1, :]  # 移除[None] token的预测
                
                # 应用温度和top-p采样
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)
                
                # 采样新token
                if top_p < 1.0:
                    # Top-p采样
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[..., 0] = False  # 保留至少一个token
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    
                    # 重新排序
                    probs = torch.zeros_like(probs)
                    probs.scatter_(-1, sorted_indices, sorted_probs)
                
                new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                new_tokens = new_tokens.view(b, t)
                
                # 决定哪些位置要更新
                mask_positions = (current_tokens == mask_token_id)
                masked_count = int(mask_positions.sum().item())
                if masked_count > 0:
                    # 每步解开一定比例的mask（至少1个）
                    num_to_unmask = max(1, int(masked_count * (1 - mask_ratio)))
                    if num_to_unmask >= masked_count:
                        # 解开全部
                        current_tokens[mask_positions] = new_tokens[mask_positions]
                    else:
                        # 随机选择要去mask的位置
                        mask_indices = torch.where(mask_positions)
                        device_idx = mask_indices[0].device
                        perm = torch.randperm(masked_count, device=device_idx)[:num_to_unmask]
                        selected_batch = mask_indices[0][perm]
                        selected_pos = mask_indices[1][perm]
                        current_tokens[selected_batch, selected_pos] = new_tokens[selected_batch, selected_pos]
        
        return current_tokens

    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())

# 创建模型的辅助函数
def create_spectrum_diffusion_model(config=None):
    """
    创建光谱扩散模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        SpectrumDiffusionModel实例
    """
    default_config = {
        'vocab_size': 84,
        'n_embd': 256,
        'n_head': 8,
        'n_layer': 6,
        'block_size': 12288,
        'cond_dim': 128
    }
    
    if config is not None:
        default_config.update(config)
    
    model = SpectrumDiffusionModel(**default_config)
    return model

if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("光谱扩散模型测试")
    print("=" * 60)
    
    # 创建模型
    model = create_spectrum_diffusion_model()
    
    # 测试前向传播
    batch_size = 2
    seq_len = 128
    x = torch.randint(1, 84, (batch_size, seq_len))  # 避免使用0（mask token）
    
    print("\n测试扩散模式前向传播:")
    logits, loss = model(x, mode='Random_CL', random_ratio=0.9)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"损失: {loss.item():.4f}")
    
    print("\n测试自回归模式前向传播:")
    logits, loss = model(x, mode='AR')
    print(f"输出形状: {logits.shape}")
    print(f"损失: {loss.item():.4f}")
    
    print("\n✅ 光谱扩散模型测试完成！")