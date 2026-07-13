#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 Transformer（与手工特征版同量级）— 固定 50 点预训练

数据: data/e2e/{train,val,test}_data.npy
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
from e2e_transformer_matched_arch import (
    D_FF,
    D_MODEL,
    DROPOUT,
    N_HEADS,
    N_LAYERS,
    MatchedTimeSeriesTransformer,
    count_parameters,
    init_weights,
)

MODEL_DIR = BASE_DIR / "train_models/e2e_transformer_50obs_matched_pretrain"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_dual_logging(MODEL_DIR, "training.log")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE_INITIAL = 1e-5
LEARNING_RATE_MIN = 1e-7
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
PATIENCE = 5
MIN_DELTA = 0.0001
USE_COSINE_ANNEALING = True
WARMUP_EPOCHS = 20
WARMUP_START_LR = 1e-7


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


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

    model = MatchedTimeSeriesTransformer(num_classes=len(classes)).to(device)
    model.apply(init_weights)
    n_params = count_parameters(model)
    logger.info(
        "Matched E2E: %dL d=%d heads=%d ff=%d | %.2fM params (~%.2f GB)",
        N_LAYERS,
        D_MODEL,
        N_HEADS,
        D_FF,
        n_params / 1e6,
        n_params * 4 / 1024**3,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_INITIAL, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    scheduler = None
    stop = EarlyStoppingTracker(patience=PATIENCE, min_delta=MIN_DELTA)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    lr_history: list[float] = []

    for epoch in range(1, N_EPOCHS + 1):
        if USE_COSINE_ANNEALING and epoch <= WARMUP_EPOCHS:
            progress = epoch / WARMUP_EPOCHS
            lr = WARMUP_START_LR + (LEARNING_RATE_INITIAL - WARMUP_START_LR) * progress
            _set_lr(opt, lr)
        elif USE_COSINE_ANNEALING and epoch == WARMUP_EPOCHS + 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=N_EPOCHS - WARMUP_EPOCHS, eta_min=LEARNING_RATE_MIN
            )
        elif scheduler is not None:
            scheduler.step()

        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                loss = torch.nan_to_num(loss)
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
    config = [
        "E2E Matched Transformer (50 obs pretrain)",
        "aligned_with: train_transformer_feat_50obs_pretrain.py",
        f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, d_ff={D_FF}, dropout={DROPOUT}",
        f"params_M={n_params/1e6:.2f}",
        f"data={BASE_DIR / 'data/e2e'}",
        f"BATCH_SIZE={BATCH_SIZE}, N_EPOCHS={N_EPOCHS}",
        f"LR={LEARNING_RATE_INITIAL}, WARMUP={WARMUP_EPOCHS}",
        f"PATIENCE={PATIENCE}, MIN_DELTA={MIN_DELTA}",
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
        config_name="training_config.txt",
        band_codes=test_band,
        lr_history=lr_history,
        band_title="Train: band_code vs accuracy",
    )


if __name__ == "__main__":
    train_loop()
