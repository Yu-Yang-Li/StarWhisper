#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 Transformer — 阶段1：固定 50 点预训练

数据: data/e2e/{train,val,test}_data.npy (3, 50)
早停: 验证集；最终评估: 测试集
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dl_common import (
    EarlyStoppingTracker,
    NumpyTensorDataset,
    evaluate_classifier_safe,
    finalize_test_outputs,
    log_epoch,
    setup_dual_logging,
)
from split_utils import BASE_DIR, load_e2e_arrays, load_e2e_label_encoder

MODEL_DIR = BASE_DIR / "train_models/transformer_e2e_model_50obs"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_dual_logging(MODEL_DIR, "training.log")
logger = logging.getLogger(__name__)

BATCH_SIZE = 128
N_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
PATIENCE = 8
MIN_DELTA = 0.0001
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1
RANDOM_STATE = 42


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 7):
        super().__init__()
        self.proj = nn.Linear(in_ch, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)
        enc = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, D_FF, DROPOUT, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h).mean(dim=1)
        return self.head(h)

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        key_pad = ~mask
        h = self.encoder(h, src_key_padding_mask=key_pad).mean(dim=1)
        return self.head(h)


def train_loop() -> None:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_e2e_arrays("train")
    X_val, y_val = load_e2e_arrays("val")
    X_test, y_test = load_e2e_arrays("test")
    le = load_e2e_label_encoder()
    classes = list(le.classes_)
    test_band = torch.round(torch.from_numpy(X_test[:, 2, 0])).numpy().astype(int)

    train_loader = DataLoader(
        NumpyTensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        NumpyTensorDataset(X_val, y_val), batch_size=256, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        NumpyTensorDataset(X_test, y_test), batch_size=256, shuffle=False, num_workers=2
    )

    model = TimeSeriesTransformer(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    stop = EarlyStoppingTracker(patience=PATIENCE, min_delta=MIN_DELTA)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    lr_history: list[float] = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        val_acc, _, _, val_loss = evaluate_classifier_safe(model, val_loader, device)
        tr_loss_avg = tr_loss / max(n, 1)
        history["train_loss"].append(tr_loss_avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        lr_history.append(opt.param_groups[0]["lr"])

        improved = stop.update(val_acc, epoch)
        if improved:
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        log_epoch(epoch, N_EPOCHS, tr_loss_avg, val_acc, lr_history[-1], stop, improved=improved)

        if stop.should_stop():
            logger.info(
                "Early stopping @ epoch %d | best val_acc=%.2f%% @ epoch %d",
                epoch,
                stop.best_acc * 100,
                stop.best_epoch,
            )
            break

    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = [
        "E2E small Transformer (50 obs pretrain)",
        f"d_model={D_MODEL}, n_layers={N_LAYERS}, params_M={n_params/1e6:.2f}",
        f"data={BASE_DIR / 'data/e2e'}",
        f"LR={LR}, PATIENCE={PATIENCE}",
        f"best_val_acc={stop.best_acc:.4f}, best_epoch={stop.best_epoch}",
    ]
    finalize_test_outputs(
        model,
        MODEL_DIR,
        RESULTS_DIR,
        test_loader,
        device,
        classes,
        history,
        le,
        config,
        band_codes=test_band,
        lr_history=lr_history,
        band_title="Train: band_code vs accuracy",
    )


if __name__ == "__main__":
    train_loop()
