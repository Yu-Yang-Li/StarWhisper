"""
EfficientNet-B0 图像分类迁移学习基线 (PyTorch)
==================================================
适用场景：图像分类迁移学习（自定义类别）；科学图像（显微/天文/医学影像/遥感）；小样本微调
数据接口：ImageFolder 目录结构 data_dir/train/class_x/xxx.jpg, data_dir/val/class_x/xxx.jpg
依赖：torch, torchvision, pillow
用法：
    python baseline_efficientnet.py --demo                     # 合成数据演示
    python baseline_efficientnet.py --data_dir ./data --num_classes 2
"""
from __future__ import annotations
import argparse, shutil
import json
from datetime import datetime
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 0. 配置 ----------
CONFIG = dict(data_dir=Path("./data"), num_classes=2, batch_size=32,
              lr_head=1e-3, lr_ft=1e-4, epochs_head=3, epochs_ft=5,
              num_workers=0, model_save_path=Path("./efficientnet_b0.pt"),
              metrics_path=Path("./metrics.json"),
              summary_path=Path("./baseline_summary.json"),
              log_path=Path("./train_log.txt"))

IMAGENET_MEAN, IMAGENET_STD = [0.485,0.456,0.406], [0.229,0.224,0.225]


def emit(message: str, log_lines: list[str]) -> None:
    print(message)
    log_lines.append(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 1. 数据 & 增广 ----------
def build_transforms():
    tr = transforms.Compose([transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    va = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tr, va

def make_loaders(data_dir, bs, nw, log_lines=None):
    tr_t, va_t = build_transforms()
    tr_ds = datasets.ImageFolder(str(data_dir/"train"), transform=tr_t)
    va_ds = datasets.ImageFolder(str(data_dir/"val"), transform=va_t)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True, num_workers=nw, pin_memory=False)
    va_ld = DataLoader(va_ds, bs, shuffle=False, num_workers=nw, pin_memory=False)
    message = f"[数据] train={len(tr_ds)} val={len(va_ds)} classes={tr_ds.classes}"
    if log_lines is None:
        print(message)
    else:
        emit(message, log_lines)
    return tr_ld, va_ld, len(tr_ds.classes)


# ---------- 2. 模型 ----------
def build_model(nc):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_f, nc))
    return model

def freeze_features(model, freeze: bool):
    for p in model.features.parameters(): p.requires_grad = not freeze
    for p in model.classifier.parameters(): p.requires_grad = True


# ---------- 3. 训练/评估 ----------
def run_epoch(model, ld, opt, crit, train):
    model.train(train); tot, cor, n = 0., 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labs in ld:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            if train: opt.zero_grad()
            logits = model(imgs); loss = crit(logits, labs)
            if train: loss.backward(); opt.step()
            tot += loss.item()*imgs.size(0)
            cor += (logits.argmax(1)==labs).sum().item(); n += imgs.size(0)
    return tot/n, cor/n

def train_phase(model, tr_ld, va_ld, epochs, lr, name, log_lines):
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss(); best = 0.
    for ep in range(1, epochs+1):
        tl, ta = run_epoch(model, tr_ld, opt, crit, True)
        vl, va = run_epoch(model, va_ld, opt, crit, False)
        emit(f"[{name}] ep {ep:>2d} train_loss={tl:.4f} acc={ta:.4f} val_loss={vl:.4f} acc={va:.4f}", log_lines)
        best = max(best, va)
    return best


# ---------- 4. demo 数据 ----------
def make_demo(data_dir, nc=2, per=60):
    rng = np.random.default_rng(42)
    colors = [rng.integers(0,255,3) for _ in range(nc)]
    if data_dir.exists(): shutil.rmtree(data_dir)
    for split in ("train","val"):
        n = per if split=="train" else per//3
        for c in range(nc):
            d = data_dir/split/f"class{c}"; d.mkdir(parents=True, exist_ok=True)
            base = colors[c]
            for i in range(n):
                arr = np.clip(base+rng.integers(-20,20,3), 0, 255).astype(np.uint8)
                Image.fromarray(np.full((64,64,3), arr, dtype=np.uint8)).save(d/f"{i}.png")


# ---------- 5. 主流程 ----------
def main():
    log_lines = []
    ap = argparse.ArgumentParser(description="EfficientNet-B0 图像分类基线")
    ap.add_argument("--data_dir", type=Path, default=CONFIG["data_dir"])
    ap.add_argument("--num_classes", type=int, default=CONFIG["num_classes"])
    ap.add_argument("--epochs_head", type=int, default=CONFIG["epochs_head"])
    ap.add_argument("--epochs_ft", type=int, default=CONFIG["epochs_ft"])
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    data_mode = "user_image_folder"
    if args.demo or not (args.data_dir/"train").exists():
        data_mode = "demo"
        emit("[demo] 自动合成 demo 数据；该结果不代表用户真实数据。", log_lines)
        make_demo(args.data_dir, args.num_classes)
    emit(f"[设备] {DEVICE}", log_lines)
    tr_ld, va_ld, auto_nc = make_loaders(args.data_dir, args.batch_size, CONFIG["num_workers"], log_lines)
    nc = args.num_classes or auto_nc
    model = build_model(nc).to(DEVICE)

    freeze_features(model, freeze=True)
    best_head = train_phase(model, tr_ld, va_ld, args.epochs_head, CONFIG["lr_head"], "head", log_lines)
    freeze_features(model, freeze=False)
    best_ft = train_phase(model, tr_ld, va_ld, args.epochs_ft, CONFIG["lr_ft"], "finetune", log_lines)

    CONFIG["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    emit(f"[保存] {CONFIG['model_save_path']}", log_lines)
    metrics = {"best_head_val_accuracy": float(best_head), "best_finetune_val_accuracy": float(best_ft)}
    summary = {
        "schema": "research_baseline_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_mode": data_mode,
        "data_source": str(args.data_dir),
        "model": "EfficientNet-B0",
        "task": "image_classification",
        "device": str(DEVICE),
        "num_classes": int(nc),
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_ft,
        "metrics": metrics,
        "model_path": str(CONFIG["model_save_path"]),
        "warning": "当前为合成 demo 图像数据，不代表用户真实数据结果。" if data_mode == "demo" else None,
    }
    write_json(CONFIG["metrics_path"], metrics)
    write_json(CONFIG["summary_path"], summary)
    emit(f"[记录] metrics={CONFIG['metrics_path']} summary={CONFIG['summary_path']} log={CONFIG['log_path']}", log_lines)
    CONFIG["log_path"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
