#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与手工特征版 Transformer (1117_50obs) 对齐的端到端时序模型定义。

配置对齐: d_model=1024, n_heads=16, n_layers=18, dim_feedforward=4096, dropout=0.1
分类头: 3 层 MLP
输入: (batch, seq_len, 3)，seq_len 最大 50
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 与 train_transformer_classifier_1117_50obs.py 一致
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 18
D_FF = 4096
DROPOUT = 0.1
MAX_SEQ_LEN = 50


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MatchedTimeSeriesTransformer(nn.Module):
    """端到端时序 Transformer，参数量 ~220–250M"""

    def __init__(self, in_ch: int = 3, num_classes: int = 7):
        super().__init__()
        self.proj = nn.Linear(in_ch, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)
        enc = nn.TransformerEncoderLayer(
            D_MODEL,
            N_HEADS,
            D_FF,
            DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, D_MODEL // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 4, num_classes),
        )

    def _encode(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=key_pad_mask)
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x, key_pad_mask=~mask))


def init_weights(m: nn.Module) -> None:
    """与特征版一致的保守 Xavier 初始化。"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.TransformerEncoderLayer):
        for name, param in m.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
