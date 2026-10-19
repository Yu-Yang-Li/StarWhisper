#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer 分类器（1117特征版）
- 特征文件（不变）: train4_1117_balanced / test4_1117_balanced（50 点）
- 划分: data/split/50obs/

说明:
- 使用 1117 全量特征（自动从CSV中读取，排除 file_path、category 两列）。
- 生成与基线脚本一致的输出：混淆矩阵、分类报告、band_code vs accuracy、
  (若存在) num_points vs accuracy、训练日志与配置、模型/编码器/标准化器。
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path("/root/shared-nvme")
from split_utils import load_split_feature_bundle, to_transformer_train_bundle  # noqa: E402

MODEL_DIR = BASE_DIR / "train_models/transformer_feat_50obs_pretrain"
RESULTS_DIR = MODEL_DIR / "results"
LOG_FILE = MODEL_DIR / "training.log"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# 训练超参
RANDOM_STATE = 42
BATCH_SIZE = 128  # 减小batch size以适应更大的模型
EVAL_BATCH_SIZE = 2048  # 减小评估batch size
N_EPOCHS = 100
LEARNING_RATE_INITIAL = 1e-5  # 初始学习率（进一步降低，防止梯度爆炸）
LEARNING_RATE_MIN = 1e-7  # 最小学习率（余弦退火终点）
WEIGHT_DECAY = 1e-4  # 降低weight decay
GRAD_CLIP_NORM = 1.0  # 梯度裁剪阈值（放宽，避免过度裁剪）
N_CLASSES_FALLBACK = 5
# Early stopping 参数
EARLY_STOPPING_PATIENCE = 5  # 容忍多少个epoch没有改善
EARLY_STOPPING_MIN_DELTA = 0.0001  # 最小改善阈值
# 余弦退火参数
USE_COSINE_ANNEALING = True
WARMUP_EPOCHS = 20  # 增加warmup阶段，更慢的学习率增长
WARMUP_START_LR = 1e-7  # Warmup起始学习率（更低，更保守）


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    classes: list
    scaler: StandardScaler
    feature_cols: list


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_heads: int = 16,
        n_layers: int = 18,
        dim_feedforward: int = 4096,
        d_model: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.pos_encoding = None
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 更大的分类头（3层）
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        h = self.input_projection(x)  # (batch, d_model)
        h = h.unsqueeze(1)  # (batch, 1, d_model)
        # 与原版保持一致：加位置编码（长度=1时影响极小，这里保留接口）
        h = self.transformer(h)  # (batch, 1, d_model)
        h = h.mean(dim=1)  # 全局平均池化 (batch, d_model)
        logits = self.classifier(h)
        return logits


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_preprocess(train_path: Path, test_path: Path) -> DataBundle:
    logger.info("加载 1117 平衡数据…")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 动态特征列：排除非特征
    exclude = {"file_path", "category", "num_points"}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    train_df = coerce_numeric(train_df, feature_cols).replace([np.inf, -np.inf], np.nan)
    test_df = coerce_numeric(test_df, feature_cols).replace([np.inf, -np.inf], np.nan)

    before_train, before_test = len(train_df), len(test_df)
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)
    logger.info(
        f"清洗: 训练丢弃 {before_train - len(train_df)}, "
        f"验证丢弃 {before_test - len(test_df)}"
    )

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df["category"].values
    y_test_raw = test_df["category"].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    classes = list(label_encoder.classes_)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 特征统计信息
    logger.info(
        f"标准化后特征统计: "
        f"均值={X_train.mean():.6f}, "
        f"标准差={X_train.std():.6f}, "
        f"min={X_train.min():.6f}, "
        f"max={X_train.max():.6f}"
    )

    # 特征值裁剪（防止极端值导致梯度爆炸）
    FEATURE_CLIP_VALUE = 5.0  # 裁剪到 [-5, 5]
    X_train = np.clip(X_train, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    X_test = np.clip(X_test, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    logger.info(
        f"特征值裁剪到 [-{FEATURE_CLIP_VALUE}, {FEATURE_CLIP_VALUE}] 后: "
        f"min={X_train.min():.6f}, max={X_train.max():.6f}"
    )

    # 非数置零(极端安全)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return DataBundle(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        classes=classes,
        scaler=scaler,
        feature_cols=feature_cols,
    )


def train_model(bundle: DataBundle):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)

    # 模型配置：目标大小 ~1GB (约 250M 参数)
    # d_model=1024, n_layers=18, dim_feedforward=4096, n_heads=16
    # 预计参数量: ~250M, 模型大小: ~1GB (float32)
    model = TransformerClassifier(
        input_dim=bundle.X_train.shape[1],
        num_classes=(len(bundle.classes) if bundle.classes else N_CLASSES_FALLBACK),
        n_heads=16,
        n_layers=18,
        dim_feedforward=4096,
        d_model=1024,
        dropout=0.1,
    ).to(device)

    # 改进的权重初始化（使用更保守的初始化）
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # 使用Xavier初始化，但添加缩放因子
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)  # 降低增益
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0.0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.TransformerEncoderLayer):
            # 对Transformer层使用更小的初始化
            for name, param in m.named_parameters():
                if "weight" in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=0.5)

    model.apply(init_weights)
    logger.info("已应用保守的Xavier权重初始化（gain=0.5）")

    # 计算并打印模型大小
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    logger.info(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    logger.info(
        f"模型大小 (float32): {model_size_mb:.2f} MB " f"({model_size_mb/1024:.2f} GB)"
    )

    # 检查类别分布
    from collections import Counter

    train_class_counts = Counter(bundle.y_train)
    test_class_counts = Counter(bundle.y_test)
    logger.info("训练集类别分布:")
    for cls_idx, cls_name in enumerate(bundle.classes):
        count = train_class_counts.get(cls_idx, 0)
        pct = count / len(bundle.y_train) * 100
        logger.info(f"  {cls_name}: {count} ({pct:.2f}%)")
    logger.info("验证集类别分布:")
    for cls_idx, cls_name in enumerate(bundle.classes):
        count = test_class_counts.get(cls_idx, 0)
        pct = count / len(bundle.y_test) * 100
        logger.info(f"  {cls_name}: {count} ({pct:.2f}%)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_INITIAL,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # 余弦退火学习率调度器
    # 注意：scheduler会在warmup后重新创建，所以这里先设为None
    scheduler = None
    if USE_COSINE_ANNEALING:
        logger.info(
            f"使用余弦退火学习率调度: "
            f"Warmup起始={WARMUP_START_LR:.2e}, "
            f"初始={LEARNING_RATE_INITIAL:.2e}, "
            f"最小={LEARNING_RATE_MIN:.2e}, "
            f"Warmup={WARMUP_EPOCHS} epochs, "
            f"余弦周期={N_EPOCHS - WARMUP_EPOCHS} epochs"
        )
    else:
        logger.info(f"使用固定学习率: {LEARNING_RATE_INITIAL:.2e}")

    def to_tensor(np_x):
        t = torch.from_numpy(np_x).to(device)
        if not torch.isfinite(t).all():
            t = torch.nan_to_num(t)
        return t

    # 创建训练数据的索引并shuffle
    train_indices = np.arange(len(bundle.X_train))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(train_indices)

    X_train_shuffled = bundle.X_train[train_indices]
    y_train_shuffled = bundle.y_train[train_indices]

    X_train_t = to_tensor(X_train_shuffled)
    y_train_t = torch.from_numpy(y_train_shuffled).long().to(device)
    X_test_t = to_tensor(bundle.X_test)
    y_test_t = torch.from_numpy(bundle.y_test).long().to(device)

    n_train = X_train_t.shape[0]
    n_batches = max(1, n_train // BATCH_SIZE + int(n_train % BATCH_SIZE > 0))

    logger.info("训练数据已shuffle")

    logger.info("开始训练 Transformer...")
    logger.info(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    best_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    lr_history = []  # 记录学习率历史

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        large_grad_count = 0  # 统计梯度范数过大的batch数量
        small_grad_count = 0  # 统计梯度范数过小的batch数量

        # 每个epoch都shuffle训练数据
        epoch_indices = np.arange(n_train)
        np.random.shuffle(epoch_indices)
        X_train_epoch = X_train_t[epoch_indices]
        y_train_epoch = y_train_t[epoch_indices]

        # Warmup阶段：线性增加学习率（从WARMUP_START_LR到LEARNING_RATE_INITIAL）
        if USE_COSINE_ANNEALING and epoch <= WARMUP_EPOCHS:
            # 线性插值：从WARMUP_START_LR到LEARNING_RATE_INITIAL
            progress = epoch / WARMUP_EPOCHS
            warmup_lr = (
                WARMUP_START_LR + (LEARNING_RATE_INITIAL - WARMUP_START_LR) * progress
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        for i in range(n_batches):
            s, e = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, n_train)
            xb = X_train_epoch[s:e]
            yb = y_train_epoch[s:e]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                loss = torch.nan_to_num(loss)
            loss.backward()
            # 进行梯度裁剪（先裁剪再检查，避免重复计算）
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )

            # 记录梯度信息（前几个epoch和batch）
            if (epoch <= 3 and i == 0) or (epoch == 1 and i < 3):
                logger.info(
                    "Epoch {} Batch {} - 梯度范数(裁剪后): {:.6f}, "
                    "Loss: {:.4f}".format(epoch, i, total_norm, loss.item())
                )
            # 统计梯度异常（不打印警告）
            if total_norm > GRAD_CLIP_NORM * 1.5:  # 裁剪后仍然较大
                large_grad_count += 1
            elif total_norm < 1e-6:
                small_grad_count += 1
            optimizer.step()
            epoch_loss += loss.item() * (e - s)

        epoch_loss /= n_train

        # 报告梯度统计信息
        if large_grad_count > 0 or small_grad_count > 0:
            logger.info(
                f"Epoch {epoch} 梯度统计: "
                f"梯度范数过大(>{GRAD_CLIP_NORM * 1.5:.2f})的batch数={large_grad_count}, "
                f"梯度范数过小(<1e-6)的batch数={small_grad_count}"
            )

        # 更新学习率（warmup之后才使用scheduler）
        if USE_COSINE_ANNEALING and scheduler is not None:
            if epoch > WARMUP_EPOCHS:
                # 调整scheduler的T_max，因为warmup占用了前几个epoch
                # 但CosineAnnealingLR的T_max是固定的，我们需要手动调整
                # 这里我们使用一个技巧：在warmup后重新创建scheduler
                if epoch == WARMUP_EPOCHS + 1:
                    # 重新创建scheduler，T_max调整为剩余epoch数
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=N_EPOCHS - WARMUP_EPOCHS,
                        eta_min=LEARNING_RATE_MIN,
                    )
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        model.eval()
        with torch.no_grad():
            # 分批评估，避免一次性大张量在GPU上导致配置/显存问题
            def batched_predict(
                x_tensor: torch.Tensor, mdl: nn.Module, bs: int
            ) -> torch.Tensor:
                preds = []
                n = x_tensor.shape[0]
                for i in range(0, n, bs):
                    xb = x_tensor[i : i + bs]
                    logits = mdl(xb)
                    preds.append(torch.argmax(logits, dim=1))
                return torch.cat(preds, dim=0)

            try:
                pred_val = batched_predict(X_test_t, model, EVAL_BATCH_SIZE)
            except RuntimeError as e:
                # CUDA 出错时，自动回退到 CPU 进行评估
                logger.warning(
                    "验证阶段出现CUDA错误，自动回退到CPU评估: %s",
                    e,
                )
                cpu_device = torch.device("cpu")
                model_cpu = model.to(cpu_device)
                X_test_cpu = X_test_t.to(cpu_device)
                pred_val = batched_predict(X_test_cpu, model_cpu, EVAL_BATCH_SIZE)
                # 评估完成后，将模型移回原设备以继续后续训练/保存
                model.to(device)

            acc = (pred_val == y_test_t).float().mean().item()

            # 检查预测分布（诊断用）
            if epoch == 1:
                from collections import Counter

                pred_counts = Counter(pred_val.cpu().numpy())
                logger.info("第一个epoch的预测分布:")
                for cls_idx, cls_name in enumerate(bundle.classes):
                    count = pred_counts.get(cls_idx, 0)
                    pct = count / len(pred_val) * 100
                    logger.info(f"  {cls_name}: {count} ({pct:.2f}%)")

            # Early stopping 逻辑
            improved = acc > (best_acc + EARLY_STOPPING_MIN_DELTA)
            if improved:
                best_acc = acc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
                logger.info(
                    f"Epoch {epoch:02d}/{N_EPOCHS} - loss={epoch_loss:.4f} - "
                    f"val_acc={acc*100:.2f}% - lr={current_lr:.2e} * "
                    "(best, saved)"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"Epoch {epoch:02d}/{N_EPOCHS} - loss={epoch_loss:.4f} - "
                    f"val_acc={acc*100:.2f}% - lr={current_lr:.2e} "
                    f"(best: {best_acc*100:.2f}% @ epoch {best_epoch}, "
                    f"patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})"
                )

            # Early stopping 检查
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"\nEarly stopping triggered at epoch {epoch}! "
                    f"Best validation accuracy: {best_acc*100:.2f}% at epoch {best_epoch}"
                )
                break

    # 保存最后模型
    torch.save(model.state_dict(), MODEL_DIR / "last_model.pth")
    logger.info(f"训练完成! 最佳验证准确率: {best_acc*100:.2f}% (epoch {best_epoch})")

    # 保存学习率历史
    if USE_COSINE_ANNEALING and lr_history:
        lr_df = pd.DataFrame(
            {"epoch": range(1, len(lr_history) + 1), "learning_rate": lr_history}
        )
        lr_df.to_csv(RESULTS_DIR / "learning_rate_history.csv", index=False)
        logger.info("学习率历史已保存到: learning_rate_history.csv")

        # 绘制学习率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(lr_df["epoch"], lr_df["learning_rate"], linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title("Learning Rate Schedule (Cosine Annealing)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / "learning_rate_schedule.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("学习率曲线已保存到: learning_rate_schedule.png")

    # 保存辅助对象
    import joblib

    joblib.dump(bundle.label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(bundle.scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(bundle.feature_cols, MODEL_DIR / "feature_columns.pkl")

    return model, device


def _predict_batched(
    x_np: np.ndarray, model: nn.Module, device: torch.device, bs: int
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(x_np).to(device)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            xb = x[i : i + bs]
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0).numpy()


def plot_confusion_and_reports(
    bundle: DataBundle, model: TransformerClassifier, device
):
    logger.info("评估并绘制混淆矩阵…")

    # 分批 + CUDA 回退
    def _predict_safe() -> np.ndarray:
        try:
            return _predict_batched(bundle.X_test, model, device, EVAL_BATCH_SIZE)
        except RuntimeError as e:
            logger.warning("评估CUDA错误，回退CPU: %s", e)
            cpu = torch.device("cpu")
            model_cpu = model.to(cpu)
            pred = _predict_batched(bundle.X_test, model_cpu, cpu, EVAL_BATCH_SIZE)
            model.to(device)
            return pred

    y_pred = _predict_safe()

    acc = accuracy_score(bundle.y_test, y_pred)
    report = classification_report(
        bundle.y_test, y_pred, target_names=bundle.classes, digits=4
    )
    cm = confusion_matrix(bundle.y_test, y_pred)

    # 保存报告
    (RESULTS_DIR / "metrics.txt").write_text(
        f"Accuracy: {acc:.4f}\n\n" + report,
        encoding="utf-8",
    )

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    import seaborn as sns

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=bundle.classes,
        yticklabels=bundle.classes,
    )
    plt.title("Confusion Matrix - Transformer (1117)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_band(bundle: DataBundle, model: TransformerClassifier, device):
    if "band_code" not in bundle.test_df.columns:
        logger.info("缺少 band_code，跳过 band_code vs accuracy")
        return
    df = bundle.test_df.copy()
    # 分批 + CUDA 回退
    try:
        pred = _predict_batched(bundle.X_test, model, device, EVAL_BATCH_SIZE)
    except RuntimeError as e:
        logger.warning("band_code图CUDA错误，回退CPU: %s", e)
        cpu = torch.device("cpu")
        model_cpu = model.to(cpu)
        pred = _predict_batched(bundle.X_test, model_cpu, cpu, EVAL_BATCH_SIZE)
        model.to(device)
    true = bundle.y_test
    le = bundle.label_encoder
    df_plot = pd.DataFrame(
        {
            "band_code": df["band_code"].values,
            "true": le.inverse_transform(true),
            "pred": le.inverse_transform(pred),
        }
    )
    g = df_plot.groupby("band_code").apply(lambda x: (x["true"] == x["pred"]).mean())
    g = g.reset_index().rename(columns={0: "accuracy"}).sort_values("band_code")
    g.to_csv(RESULTS_DIR / "band_code_vs_accuracy.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(g["band_code"], g["accuracy"], marker="o")
    plt.xlabel("band_code")
    plt.ylabel("accuracy")
    plt.title("Validation: band_code vs accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "band_code_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_accuracy_by_numpoints(
    bundle: DataBundle, model: TransformerClassifier, device
):
    if "num_points" not in bundle.test_df.columns:
        logger.info("缺少 num_points，跳过 num_points vs accuracy")
        return
    df = bundle.test_df.copy()
    # 分批 + CUDA 回退
    try:
        pred = _predict_batched(bundle.X_test, model, device, EVAL_BATCH_SIZE)
    except RuntimeError as e:
        logger.warning("num_points图CUDA错误，回退CPU: %s", e)
        cpu = torch.device("cpu")
        model_cpu = model.to(cpu)
        pred = _predict_batched(bundle.X_test, model_cpu, cpu, EVAL_BATCH_SIZE)
        model.to(device)
    true = bundle.y_test
    le = bundle.label_encoder
    df_plot = pd.DataFrame(
        {
            "num_points": df["num_points"].values,
            "true": le.inverse_transform(true),
            "pred": le.inverse_transform(pred),
        }
    )
    g = df_plot.groupby("num_points").apply(lambda x: (x["true"] == x["pred"]).mean())
    g = g.reset_index().rename(columns={0: "accuracy"}).sort_values("num_points")
    g.to_csv(RESULTS_DIR / "num_points_vs_accuracy.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(g["num_points"], g["accuracy"], marker="o")
    plt.xlabel("num_points")
    plt.ylabel("accuracy")
    plt.title("Validation: num_points vs accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "num_points_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_config(bundle: DataBundle):
    cfg_path = MODEL_DIR / "training_config.txt"
    from split_utils import FEATURE_POOLS  # noqa: E402

    pool = FEATURE_POOLS["50obs"]
    lines = [
        "Transformer (1117, 50obs) 训练配置\n",
        f"FEATURE_POOL: 50obs\n",
        f"TRAIN_CSV: {pool['train_csv']}\n",
        f"TEST_CSV: {pool['test_csv']}\n",
        f"SPLIT_DIR: {BASE_DIR / 'data/split/50obs'}\n",
        f"N_EPOCHS: {N_EPOCHS}\n",
        f"BATCH_SIZE: {BATCH_SIZE}\n",
        f"LEARNING_RATE_INITIAL: {LEARNING_RATE_INITIAL}\n",
        f"LEARNING_RATE_MIN: {LEARNING_RATE_MIN}\n",
        f"USE_COSINE_ANNEALING: {USE_COSINE_ANNEALING}\n",
        f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n",
        f"WARMUP_START_LR: {WARMUP_START_LR}\n",
        f"WEIGHT_DECAY: {WEIGHT_DECAY}\n",
        f"GRAD_CLIP_NORM: {GRAD_CLIP_NORM}\n",
        f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}\n",
        f"EARLY_STOPPING_MIN_DELTA: {EARLY_STOPPING_MIN_DELTA}\n",
        "MODEL_CONFIG: d_model=1024, n_layers=18, "
        "dim_feedforward=4096, n_heads=16\n",
        f"FEATURE_COLUMNS: {', '.join(bundle.feature_cols)}\n",
    ]
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info(f"训练配置已保存: {cfg_path}")


def predict_extra_dataset(
    extra_test_file: Path,
    bundle: DataBundle,
    model: TransformerClassifier,
    device: torch.device,
):
    """对额外的测试集进行预测并输出结果"""
    logger.info(f"\n{'='*80}")
    logger.info(f"额外预测数据集: {extra_test_file}")
    logger.info(f"{'='*80}")

    # 加载额外测试集
    extra_df = pd.read_csv(extra_test_file)
    logger.info(f"加载额外测试集: {len(extra_df)} 个样本")

    # 使用训练时的特征列
    feature_cols = bundle.feature_cols
    available_feature_cols = [c for c in feature_cols if c in extra_df.columns]

    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        logger.warning(f"缺少特征列: {missing}")
        logger.warning(
            f"使用可用特征: {len(available_feature_cols)}/{len(feature_cols)}"
        )

    # 数据预处理（与训练时一致）
    extra_df = coerce_numeric(extra_df, available_feature_cols).replace(
        [np.inf, -np.inf], np.nan
    )
    before_len = len(extra_df)
    extra_df = extra_df.dropna(subset=available_feature_cols)
    logger.info(f"清洗: 丢弃 {before_len - len(extra_df)} 个样本")

    # 确保特征列顺序一致
    X_extra = extra_df[available_feature_cols].values.astype(np.float32)

    # 使用训练时的scaler进行标准化
    X_extra = bundle.scaler.transform(X_extra)

    # 特征值裁剪
    FEATURE_CLIP_VALUE = 5.0
    X_extra = np.clip(X_extra, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)

    # 非数置零
    X_extra = np.nan_to_num(X_extra, nan=0.0, posinf=0.0, neginf=0.0)

    # 获取真实标签
    if "category" not in extra_df.columns:
        logger.error("额外测试集缺少category列，无法计算准确率")
        return

    y_extra_raw = extra_df["category"].values
    # 使用训练时的label_encoder
    try:
        y_extra = bundle.label_encoder.transform(y_extra_raw)
    except ValueError as e:
        logger.warning(f"标签编码错误: {e}，尝试处理未知标签")
        # 处理未知标签：映射到已知类别或跳过
        known_classes = set(bundle.label_encoder.classes_)
        mask = [label in known_classes for label in y_extra_raw]
        if not any(mask):
            logger.error("没有已知类别的样本，无法进行预测")
            return
        extra_df = extra_df[mask]
        X_extra = X_extra[mask]
        y_extra_raw = y_extra_raw[mask]
        y_extra = bundle.label_encoder.transform(y_extra_raw)
        logger.info(f"过滤后样本数: {len(extra_df)}")

    # 预测
    logger.info("开始预测...")
    try:
        y_pred = _predict_batched(X_extra, model, device, EVAL_BATCH_SIZE)
    except RuntimeError as e:
        logger.warning(f"预测CUDA错误，回退CPU: {e}")
        cpu = torch.device("cpu")
        model_cpu = model.to(cpu)
        y_pred = _predict_batched(X_extra, model_cpu, cpu, EVAL_BATCH_SIZE)
        model.to(device)

    # 计算指标
    acc = accuracy_score(y_extra, y_pred)
    report = classification_report(
        y_extra, y_pred, target_names=bundle.classes, digits=4
    )
    cm = confusion_matrix(y_extra, y_pred)

    logger.info(f"\n额外测试集准确率: {acc:.4f}")
    logger.info(f"\n分类报告:\n{report}")

    # 保存结果
    extra_results_dir = RESULTS_DIR / "extra_prediction"
    extra_results_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    (extra_results_dir / "metrics.txt").write_text(
        f"Accuracy: {acc:.4f}\n\n" + report,
        encoding="utf-8",
    )

    # 保存混淆矩阵
    plt.figure(figsize=(12, 10))
    import seaborn as sns

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=bundle.classes,
        yticklabels=bundle.classes,
    )
    plt.title(f"Confusion Matrix - Extra Test Set ({extra_test_file.name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        extra_results_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 保存预测结果CSV
    pred_df = extra_df.copy()
    pred_df["true_label"] = y_extra_raw
    pred_df["predicted_label"] = bundle.label_encoder.inverse_transform(y_pred)
    pred_df["correct"] = (y_extra == y_pred).astype(int)

    # 只保留关键列
    output_cols = [
        "file_path",
        "category",
        "true_label",
        "predicted_label",
        "correct",
    ]
    if "band_code" in pred_df.columns:
        output_cols.append("band_code")
    if "num_points" in pred_df.columns:
        output_cols.append("num_points")

    pred_df[output_cols].to_csv(
        extra_results_dir / "predictions.csv",
        index=False,
        encoding="utf-8",
    )

    # 绘制 num_points vs accuracy
    if "num_points" in extra_df.columns:
        logger.info("绘制 num_points vs accuracy...")
        df_plot = pd.DataFrame(
            {
                "num_points": extra_df["num_points"].values,
                "true": y_extra_raw,
                "pred": bundle.label_encoder.inverse_transform(y_pred),
            }
        )
        g = df_plot.groupby("num_points").apply(
            lambda x: (x["true"] == x["pred"]).mean()
        )
        g = g.reset_index().rename(columns={0: "accuracy"}).sort_values("num_points")
        g.to_csv(extra_results_dir / "num_points_vs_accuracy.csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.plot(g["num_points"], g["accuracy"], marker="o")
        plt.xlabel("num_points")
        plt.ylabel("accuracy")
        plt.title(
            f"Extra Test Set: num_points vs accuracy " f"({extra_test_file.name})"
        )
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            extra_results_dir / "num_points_vs_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("✅ num_points vs accuracy 图表已保存")
    else:
        logger.info("缺少 num_points 列，跳过 num_points vs accuracy")

    logger.info(f"✅ 额外预测结果已保存到: {extra_results_dir}")
    logger.info("  - 混淆矩阵: confusion_matrix.png")
    logger.info("  - 分类报告: metrics.txt")
    logger.info("  - 预测结果: predictions.csv")
    if "num_points" in extra_df.columns:
        logger.info("  - num_points vs accuracy: num_points_vs_accuracy.png")


def main():
    split_b = load_split_feature_bundle(pool="50obs")
    bundle, split_b = to_transformer_train_bundle(split_b)
    save_config(bundle)
    model, device = train_model(bundle)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    test_bundle = DataBundle(
        train_df=split_b.train_df,
        test_df=split_b.test_df,
        X_train=split_b.X_train,
        y_train=split_b.y_train,
        X_test=split_b.X_test,
        y_test=split_b.y_test,
        label_encoder=split_b.label_encoder,
        classes=split_b.classes,
        scaler=split_b.scaler,
        feature_cols=split_b.feature_cols,
    )
    plot_confusion_and_reports(test_bundle, model, device)
    plot_accuracy_by_band(test_bundle, model, device)
    plot_accuracy_by_numpoints(test_bundle, model, device)
    logger.info(f"全部结果已输出到: {MODEL_DIR}")


if __name__ == "__main__":
    main()
