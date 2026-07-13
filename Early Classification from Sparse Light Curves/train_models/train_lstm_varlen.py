#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 2 层 LSTM：原始序列 3–30 点变长序列从头训练。

数据: data/e2e_varlen/（与 finetune_e2e_transformer_varlen.py 相同划分）
早停: 验证集；最终评估: 测试集
"""

from __future__ import annotations

import logging
import pickle

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
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

VARLEN_DIR = BASE_DIR / "data/e2e_varlen"
MODEL_DIR = BASE_DIR / "train_models/lstm_varlen"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

setup_dual_logging(MODEL_DIR, "training.log")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
BATCH_SIZE = 128
N_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 0.0
HIDDEN = 128
PATIENCE = 10
MIN_DELTA = 0.0001
GRAD_CLIP_NORM = 1.0
MAX_LEN = 30


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


class LSTMClassifier(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128, num_classes: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu().to(torch.int64)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_labels, _ = load_varlen_data("train")
    val_data, val_labels, _ = load_varlen_data("val")
    test_data, test_labels, test_lengths = load_varlen_data("test")
    test_band = extract_band_codes_from_varlen(test_data)

    le_path = VARLEN_DIR / "label_encoder.pkl"
    if not le_path.is_file():
        raise FileNotFoundError(f"缺少 {le_path}，请先运行: python3 data/prepare_e2e_varlen.py")
    le = joblib.load(le_path)
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

    model = LSTMClassifier(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    stop = EarlyStoppingTracker(patience=PATIENCE, min_delta=MIN_DELTA)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    lr_history: list[float] = []

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("RNN E2E varlen: hidden=%d, params=%.2fM", HIDDEN, n_params / 1e6)

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
        log_epoch(epoch, N_EPOCHS, tr_loss_avg, val_acc, lr_history[-1], stop, improved=improved, phase="RNN")

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
        "RNN LSTM E2E (3-30 obs, from scratch)",
        f"data={VARLEN_DIR}",
        f"HIDDEN={HIDDEN}, BATCH_SIZE={BATCH_SIZE}, LR={LR}",
        f"PATIENCE={PATIENCE}, params_M={n_params/1e6:.2f}",
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
        num_points_title="RNN: num_points vs accuracy",
        band_title="RNN: band_code vs accuracy",
    )


if __name__ == "__main__":
    main()
