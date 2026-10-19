#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 Transformer — 阶段2：真实 3–30 点微调（使用 train2 原始变长数据）

数据来源: prepare_e2e_varlen.py 生成的 e2e_varlen/ 目录
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
from train_e2e_transformer_50obs_pretrain import TimeSeriesTransformer

PRETRAINED_DIR = BASE_DIR / "train_models/e2e_transformer_50obs_pretrain"
MODEL_DIR = BASE_DIR / "train_models/e2e_transformer_varlen_finetune"
RESULTS_DIR = MODEL_DIR / "results"
VARLEN_DIR = BASE_DIR / "data/e2e_varlen"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_dual_logging(MODEL_DIR, "finetuning.log")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
BATCH_SIZE = 128
N_EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
PATIENCE = 6
MIN_DELTA = 0.0001
MAX_LEN = 50


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


def train_finetune() -> None:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = TimeSeriesTransformer(num_classes=len(classes)).to(device)
    model.load_state_dict(
        torch.load(PRETRAINED_DIR / "best_model.pth", map_location=device), strict=True
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    stop = EarlyStoppingTracker(patience=PATIENCE, min_delta=MIN_DELTA)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    lr_history: list[float] = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb, mask in train_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model.forward_with_mask(xb, mask), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        val_acc, _, _, val_loss = evaluate_classifier_safe(model, val_loader, device, use_mask=True)
        tr_loss_avg = tr_loss / max(n, 1)
        history["train_loss"].append(tr_loss_avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        lr_history.append(opt.param_groups[0]["lr"])

        improved = stop.update(val_acc, epoch)
        if improved:
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        log_epoch(
            epoch, N_EPOCHS, tr_loss_avg, val_acc, lr_history[-1], stop,
            improved=improved, phase="Finetune",
        )
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
        "E2E small Transformer finetune (3-30 obs)",
        f"pretrained={PRETRAINED_DIR}",
        f"data={VARLEN_DIR}",
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
        config_name="finetuning_config.txt",
        use_mask=True,
        test_lengths=test_lengths,
        band_codes=test_band,
        lr_history=lr_history,
    )


if __name__ == "__main__":
    train_finetune()
