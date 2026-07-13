#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 Matched Transformer — 3–30 点变长 **从头训练**（无 50obs 预训练）

消融用途：与 finetune_e2e_transformer_varlen_matched.py 对照，量化 50 点预训练贡献。
数据: data/e2e_varlen/（与微调版相同划分）
"""

from __future__ import annotations

import logging
import pickle

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dl_common import (
    EarlyStoppingTracker,
    TrueVarLenDataset,
    evaluate_classifier_safe,
    extract_band_codes_from_varlen,
    finalize_test_outputs,
    log_epoch,
    setup_dual_logging,
)
from split_utils import BASE_DIR
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

MODEL_DIR = BASE_DIR / "train_models/e2e_transformer_varlen_matched_scratch"
RESULTS_DIR = MODEL_DIR / "results"
VARLEN_DIR = BASE_DIR / "data/e2e_varlen"
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
MAX_LEN = 50
USE_COSINE_ANNEALING = True
WARMUP_EPOCHS = 20
WARMUP_START_LR = 1e-7


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def load_varlen_data(split_name: str):
    with open(VARLEN_DIR / f"{split_name}_data.pkl", "rb") as f:
        data_list = pickle.load(f)
    labels = np.load(VARLEN_DIR / f"{split_name}_labels.npy")
    lengths = np.load(VARLEN_DIR / f"{split_name}_lengths.npy")
    logger.info(
        "%s: %d 样本, 长度 %d–%d",
        split_name,
        len(data_list),
        int(lengths.min()),
        int(lengths.max()),
    )
    return data_list, labels, lengths


def train_scratch() -> None:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    train_data, train_labels, _ = load_varlen_data("train")
    val_data, val_labels, _ = load_varlen_data("val")
    test_data, test_labels, test_lengths = load_varlen_data("test")
    test_band = extract_band_codes_from_varlen(test_data)

    le = joblib.load(VARLEN_DIR / "label_encoder.pkl")
    classes = list(le.classes_)

    train_loader = DataLoader(
        TrueVarLenDataset(train_data, train_labels, max_len=MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TrueVarLenDataset(val_data, val_labels, max_len=MAX_LEN),
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        TrueVarLenDataset(test_data, test_labels, max_len=MAX_LEN),
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    model = MatchedTimeSeriesTransformer(num_classes=len(classes)).to(device)
    model.apply(init_weights)
    n_params = count_parameters(model)
    logger.info(
        "Matched E2E scratch (varlen): %dL d=%d heads=%d ff=%d | %.2fM params (~%.2f GB)",
        N_LAYERS,
        D_MODEL,
        N_HEADS,
        D_FF,
        n_params / 1e6,
        n_params * 4 / 1024**3,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_INITIAL, weight_decay=WEIGHT_DECAY
    )
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
        for xb, yb, mask in train_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model.forward_with_mask(xb, mask), yb)
            if not torch.isfinite(loss):
                loss = torch.nan_to_num(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        val_acc, _, _, val_loss = evaluate_classifier_safe(
            model, val_loader, device, use_mask=True
        )
        tr_loss_avg = tr_loss / max(n, 1)
        history["train_loss"].append(tr_loss_avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        lr_history.append(opt.param_groups[0]["lr"])

        improved = stop.update(val_acc, epoch)
        if improved:
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        log_epoch(
            epoch,
            N_EPOCHS,
            tr_loss_avg,
            val_acc,
            lr_history[-1],
            stop,
            improved=improved,
            phase="Train",
        )
        if stop.should_stop():
            logger.info(
                "Early stopping @ epoch %d | best val_acc=%.2f%% @ epoch %d",
                epoch,
                stop.best_acc * 100,
                stop.best_epoch,
            )
            break

    best_ckpt = MODEL_DIR / "best_model.pth"
    if not best_ckpt.is_file():
        raise RuntimeError("训练结束但未保存 best_model.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    config = [
        "E2E Matched Transformer scratch (3-30 obs, no 50obs pretrain)",
        "ablation_vs: finetune_e2e_transformer_varlen_matched.py",
        f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, d_ff={D_FF}, dropout={DROPOUT}",
        f"params_M={n_params/1e6:.2f}",
        f"data={VARLEN_DIR}",
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
        use_mask=True,
        test_lengths=test_lengths,
        band_codes=test_band,
        lr_history=lr_history,
        num_points_title="Scratch: num_points vs accuracy",
        band_title="Scratch: band_code vs accuracy",
    )


if __name__ == "__main__":
    train_scratch()
