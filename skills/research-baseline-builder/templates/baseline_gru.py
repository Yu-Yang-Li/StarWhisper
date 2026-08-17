"""
GRU 时序预测基线模板 (PyTorch)
===================================
适用场景：单变量/多变量时序预测（传感器、金融、气象、实验监控），过去N步→未来1/多步
数据接口：series 为 numpy.ndarray (T,) / (T, F)，或 pandas.Series/DataFrame
依赖：torch, numpy, pandas, scikit-learn
用法：python baseline_gru.py  # 用合成正弦序列跑演示
"""
from __future__ import annotations

import argparse, math
import json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 0. 配置 ----------
CONFIG = dict(seq_len=24, horizon=1, hidden_size=64, num_layers=2, dropout=0.2,
              batch_size=64, lr=1e-3, epochs=30, train_ratio=0.8,
              model_save_path=Path("./gru_model.pt"),
              metrics_path=Path("./metrics.json"),
              summary_path=Path("./baseline_summary.json"),
              log_path=Path("./train_log.txt"))


def emit(message: str, log_lines: list[str]) -> None:
    print(message)
    log_lines.append(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 1. 数据集 ----------
class TimeWindowDataset(Dataset):
    def __init__(self, data, seq_len, horizon):
        d = torch.FloatTensor(data)
        self.data = d if d.dim() == 2 else d.unsqueeze(-1)
        self.seq_len, self.horizon = seq_len, horizon
    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1
    def __getitem__(self, i):
        x = self.data[i:i+self.seq_len]
        y = self.data[i+self.seq_len:i+self.seq_len+self.horizon, 0]
        return x, y


# ---------- 2. 模型 ----------
class GRUForecaster(nn.Module):
    def __init__(self, in_f, hidden, n_layers, horizon, dropout):
        super().__init__()
        self.gru = nn.GRU(in_f, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ---------- 3. 训练/评估 ----------
def make_loaders(series, cfg):
    arr = series.values.astype(np.float32) if hasattr(series, "values") else np.asarray(series, dtype=np.float32)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    n_tr = int(len(arr) * cfg["train_ratio"])
    scaler = StandardScaler().fit(arr[:n_tr])
    arr_s = scaler.transform(arr)
    tr_arr, te_arr = arr_s[:n_tr], arr_s[n_tr - cfg["seq_len"]:]
    tr_ds = TimeWindowDataset(tr_arr, cfg["seq_len"], cfg["horizon"])
    te_ds = TimeWindowDataset(te_arr, cfg["seq_len"], cfg["horizon"])
    tr_ld = DataLoader(tr_ds, cfg["batch_size"], shuffle=True, drop_last=True)
    te_ld = DataLoader(te_ds, cfg["batch_size"], shuffle=False)
    return tr_ld, te_ld, scaler, arr.shape[1]

def train_epoch(model, ld, opt, crit):
    model.train(); tot = 0.0
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(ld.dataset)

@torch.no_grad()
def evaluate(model, ld, crit, scaler):
    model.eval(); ps, ts = [], []; tot = 0.0
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x); tot += crit(p, y).item() * x.size(0)
        ps.append(p.cpu().numpy()); ts.append(y.cpu().numpy())
    ps, ts = np.concatenate(ps), np.concatenate(ts)
    s0, m0 = scaler.scale_[0], scaler.mean_[0]
    pi, ti = ps * s0 + m0, ts * s0 + m0
    mae = mean_absolute_error(ti.reshape(-1), pi.reshape(-1))
    rmse = math.sqrt(mean_squared_error(ti.reshape(-1), pi.reshape(-1)))
    return tot / len(ld.dataset), mae, rmse


# ---------- 4. 主流程 ----------
def run(series, cfg, data_mode="demo", source=None):
    log_lines = []
    emit(f"[数据模式] {data_mode}：{'当前使用合成正弦序列，不代表用户真实数据结果。' if data_mode == 'demo' else '当前使用用户提供的数据文件。'}", log_lines)
    tr_ld, te_ld, scaler, in_f = make_loaders(series, cfg)
    model = GRUForecaster(in_f, cfg["hidden_size"], cfg["num_layers"],
                          cfg["horizon"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    final_metrics = {}
    for ep in range(1, cfg["epochs"]+1):
        tl = train_epoch(model, tr_ld, opt, crit)
        vl, mae, rmse = evaluate(model, te_ld, crit, scaler)
        final_metrics = {"val_loss": float(vl), "mae": float(mae), "rmse": float(rmse)}
        if ep == 1 or ep % max(1, cfg["epochs"]//5) == 0:
            emit(f"Epoch {ep:>3d} train={tl:.5f} val={vl:.5f} MAE={mae:.4f} RMSE={rmse:.4f}", log_lines)
    cfg["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "scaler": scaler}, cfg["model_save_path"])
    emit(f"[保存] {cfg['model_save_path']}  device={DEVICE}", log_lines)
    summary = {
        "schema": "research_baseline_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_mode": data_mode,
        "data_source": str(source) if source else None,
        "model": "GRU",
        "task": "time_series_forecast",
        "device": str(DEVICE),
        "series_length": int(len(series)),
        "seq_len": cfg["seq_len"],
        "horizon": cfg["horizon"],
        "epochs": cfg["epochs"],
        "metrics": final_metrics,
        "model_path": str(cfg["model_save_path"]),
        "warning": "当前为合成 demo 数据，不代表用户真实数据结果。" if data_mode == "demo" else None,
    }
    write_json(cfg["metrics_path"], final_metrics)
    write_json(cfg["summary_path"], summary)
    emit(f"[记录] metrics={cfg['metrics_path']} summary={cfg['summary_path']} log={cfg['log_path']}", log_lines)
    cfg["log_path"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return model, summary

def load_demo(n=2000):
    t = np.linspace(0, 20*np.pi, n, dtype=np.float32)
    return pd.Series(np.sin(t) + 0.1*np.random.randn(n).astype(np.float32), name="signal")


def load_csv(csv_path: Path, value_column: str):
    data = pd.read_csv(csv_path)
    if value_column not in data.columns:
        raise ValueError(f"value column not found: {value_column}")
    return pd.Series(data[value_column].astype(float).to_numpy(), name=value_column)

def main():
    ap = argparse.ArgumentParser(description="GRU 时序基线")
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--seq_len", type=int, default=CONFIG["seq_len"])
    ap.add_argument("--csv", type=Path, help="可选：用户 CSV 时序数据路径。")
    ap.add_argument("--value-column", help="CSV 中作为预测目标的数值列。")
    args = ap.parse_args()
    cfg = {**CONFIG, "epochs": args.epochs, "seq_len": args.seq_len}
    print(f"[设备] {DEVICE}")
    if args.csv:
        if not args.value_column:
            ap.error("--csv requires --value-column")
        run(load_csv(args.csv, args.value_column), cfg, data_mode="user_csv", source=args.csv)
    else:
        run(load_demo(), cfg, data_mode="demo")

if __name__ == "__main__":
    main()
