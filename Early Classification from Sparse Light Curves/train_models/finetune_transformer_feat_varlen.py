#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer 分类器微调脚本（基于1117_50obs模型）
- 预训练模型: transformer_feat_50obs_pretrain
- 训练集: /root/shared-nvme/features/train2_1117_20251117_235357_balanced.csv
- 验证集: /root/shared-nvme/features/test2_1117_20251117_235357_balanced.csv

说明:
- 加载预训练模型进行微调
- 使用较小的学习率进行微调
- 保持相同的模型架构和特征处理方式
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
# 预训练模型目录
PRETRAINED_MODEL_DIR = BASE_DIR / "train_models/transformer_feat_50obs_pretrain"
# 微调数据集
from split_utils import load_split_feature_bundle, to_transformer_train_bundle  # noqa: E402

# 微调后的模型保存目录
FINETUNED_MODEL_DIR = (
    BASE_DIR / "train_models/transformer_feat_varlen_finetune"
)
RESULTS_DIR = FINETUNED_MODEL_DIR / "results"
LOG_FILE = FINETUNED_MODEL_DIR / "finetuning.log"

FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
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


# 微调超参（使用更小的学习率）
RANDOM_STATE = 42
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 2048
N_EPOCHS = 50  # 微调通常需要更少的epoch
LEARNING_RATE_INITIAL = 1e-6  # 微调使用更小的学习率
LEARNING_RATE_MIN = 1e-8
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
N_CLASSES_FALLBACK = 5
# Early stopping 参数
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0001
# 余弦退火参数
USE_COSINE_ANNEALING = True
WARMUP_EPOCHS = 10  # 微调时减少warmup
WARMUP_START_LR = 1e-8


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
        h = self.transformer(h)  # (batch, 1, d_model)
        h = h.mean(dim=1)  # 全局平均池化 (batch, d_model)
        logits = self.classifier(h)
        return logits


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_preprocess(train_path: Path, test_path: Path) -> DataBundle:
    logger.info("加载微调数据集…")
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

    # 特征值裁剪
    FEATURE_CLIP_VALUE = 5.0
    X_train = np.clip(X_train, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    X_test = np.clip(X_test, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    logger.info(
        f"特征值裁剪到 [-{FEATURE_CLIP_VALUE}, {FEATURE_CLIP_VALUE}] 后: "
        f"min={X_train.min():.6f}, max={X_train.max():.6f}"
    )

    # 非数置零
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


def load_pretrained_model(
    bundle: DataBundle, pretrained_dir: Path
) -> tuple[nn.Module, torch.device]:
    """加载预训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 检查预训练模型文件
    best_model_path = pretrained_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"预训练模型文件不存在: {best_model_path}")

    # 加载预训练的scaler和label_encoder（用于特征对齐）
    import joblib

    pretrained_feature_cols_path = pretrained_dir / "feature_columns.pkl"

    if pretrained_feature_cols_path.exists():
        pretrained_feature_cols = joblib.load(pretrained_feature_cols_path)
        logger.info(
            f"预训练模型特征数: {len(pretrained_feature_cols)}, "
            f"当前数据特征数: {len(bundle.feature_cols)}"
        )
        # 检查特征是否匹配
        if set(pretrained_feature_cols) != set(bundle.feature_cols):
            logger.warning("特征列不完全匹配，将使用当前数据的特征列")

    # 创建模型（使用当前数据的特征数和类别数）
    model = TransformerClassifier(
        input_dim=bundle.X_train.shape[1],
        num_classes=len(bundle.classes),
        n_heads=16,
        n_layers=18,
        dim_feedforward=4096,
        d_model=1024,
        dropout=0.1,
    ).to(device)

    # 加载预训练权重
    logger.info(f"加载预训练模型: {best_model_path}")
    pretrained_state = torch.load(best_model_path, map_location=device)

    # 尝试加载权重（处理可能的维度不匹配）
    model_state = model.state_dict()
    pretrained_state_filtered = {}

    for key, value in pretrained_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                pretrained_state_filtered[key] = value
            else:
                logger.warning(
                    f"跳过维度不匹配的层: {key} "
                    f"(模型: {model_state[key].shape}, "
                    f"预训练: {value.shape})"
                )
        else:
            logger.warning(f"跳过不存在的层: {key}")

    model_state.update(pretrained_state_filtered)
    model.load_state_dict(model_state, strict=False)
    logger.info(f"成功加载 {len(pretrained_state_filtered)}/{len(pretrained_state)} 层")

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")

    return model, device


def finetune_model(bundle: DataBundle, model: nn.Module, device: torch.device):
    """微调模型"""
    logger.info("开始微调...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_INITIAL,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # 余弦退火学习率调度器
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

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    lr_history = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # 训练阶段
        for i in range(0, len(X_train_t), BATCH_SIZE):
            xb = X_train_t[i : i + BATCH_SIZE]
            yb = y_train_t[i : i + BATCH_SIZE]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        # 学习率调度
        if USE_COSINE_ANNEALING:
            if epoch <= WARMUP_EPOCHS:
                # Warmup阶段
                current_lr = WARMUP_START_LR + (
                    LEARNING_RATE_INITIAL - WARMUP_START_LR
                ) * (epoch / WARMUP_EPOCHS)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            else:
                # 余弦退火阶段
                if scheduler is None:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=N_EPOCHS - WARMUP_EPOCHS,
                        eta_min=LEARNING_RATE_MIN,
                    )
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = LEARNING_RATE_INITIAL

        lr_history.append(current_lr)

        # 验证阶段
        model.eval()
        with torch.no_grad():

            def batched_predict(
                x_tensor: torch.Tensor,
                mdl: nn.Module,
                bs: int,
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
                logger.warning(f"验证阶段CUDA错误，回退CPU: {e}")
                cpu_device = torch.device("cpu")
                model_cpu = model.to(cpu_device)
                X_test_cpu = X_test_t.to(cpu_device)
                pred_val = batched_predict(X_test_cpu, model_cpu, EVAL_BATCH_SIZE)
                model.to(device)

            acc = (pred_val == y_test_t).float().mean().item()

            # Early stopping 逻辑
            improved = acc > (best_acc + EARLY_STOPPING_MIN_DELTA)
            if improved:
                best_acc = acc
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    FINETUNED_MODEL_DIR / "best_model.pth",
                )
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
                    f"Best validation accuracy: {best_acc*100:.2f}% "
                    f"at epoch {best_epoch}"
                )
                break

    # 保存最后模型
    torch.save(model.state_dict(), FINETUNED_MODEL_DIR / "last_model.pth")
    logger.info(f"微调完成! 最佳验证准确率: {best_acc*100:.2f}% (epoch {best_epoch})")

    # 保存学习率历史
    if USE_COSINE_ANNEALING and lr_history:
        lr_df = pd.DataFrame(
            {
                "epoch": range(1, len(lr_history) + 1),
                "learning_rate": lr_history,
            }
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

    # 保存辅助对象
    import joblib

    joblib.dump(bundle.label_encoder, FINETUNED_MODEL_DIR / "label_encoder.pkl")
    joblib.dump(bundle.scaler, FINETUNED_MODEL_DIR / "scaler.pkl")
    joblib.dump(bundle.feature_cols, FINETUNED_MODEL_DIR / "feature_columns.pkl")

    return model, device


def _predict_batched(
    x_np: np.ndarray,
    model: nn.Module,
    device: torch.device,
    bs: int,
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
    bundle: DataBundle,
    model: TransformerClassifier,
    device,
):
    logger.info("评估并绘制混淆矩阵…")

    def _predict_safe() -> np.ndarray:
        try:
            return _predict_batched(bundle.X_test, model, device, EVAL_BATCH_SIZE)
        except RuntimeError as e:
            logger.warning(f"评估CUDA错误，回退CPU: {e}")
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
    plt.title("Confusion Matrix - Finetuned Transformer (1117)")
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
    plt.title("Finetuned: band_code vs accuracy")
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
    plt.title("Finetuned: num_points vs accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "num_points_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_config(bundle: DataBundle):
    cfg_path = FINETUNED_MODEL_DIR / "finetuning_config.txt"
    from split_utils import FEATURE_POOLS  # noqa: E402

    pool = FEATURE_POOLS["varlen"]
    lines = [
        "Transformer (1117) 微调配置\n",
        f"预训练模型: {PRETRAINED_MODEL_DIR}\n",
        f"FEATURE_POOL: varlen\n",
        f"TRAIN_CSV: {pool['train_csv']}\n",
        f"TEST_CSV: {pool['test_csv']}\n",
        f"SPLIT_DIR: {BASE_DIR / 'data/split/varlen'}\n",
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
    logger.info(f"微调配置已保存: {cfg_path}")


def main():
    logger.info("=" * 80)
    logger.info("【Transformer 分类器微调 - 基于1117_50obs模型】")
    logger.info(f"预训练模型: {PRETRAINED_MODEL_DIR}")
    logger.info("数据: data/split + features_1117（合并类别）")
    logger.info("=" * 80)

    split_b = load_split_feature_bundle(pool="varlen")
    bundle, split_b = to_transformer_train_bundle(split_b)
    save_config(bundle)

    # 加载预训练模型
    model, device = load_pretrained_model(bundle, PRETRAINED_MODEL_DIR)

    # 微调模型
    model, device = finetune_model(bundle, model, device)

    # 载入最佳模型评估
    model.load_state_dict(
        torch.load(FINETUNED_MODEL_DIR / "best_model.pth", map_location=device)
    )
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

    logger.info(f"全部结果已输出到: {FINETUNED_MODEL_DIR}")


if __name__ == "__main__":
    main()
