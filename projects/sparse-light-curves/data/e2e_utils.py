"""Shared CSV lookup and band encoding for fixed and variable-length curves."""

from pathlib import Path

import pandas as pd


FILTERCODE_MAP = {"zg": 1, "zr": 2, "zz": 3, "g": 1, "r": 2, "z": 3}


def resolve_csv_path(file_path: str, base_dir: Path, *, fallback_pool: str | None = None) -> Path | None:
    recorded = str(file_path).replace("\\", "/")
    original = Path(recorded).expanduser()
    if original.is_absolute() and original.is_file():
        return original

    parts = recorded.strip("/").split("/")
    # Old manifests may carry the previous machine's root. Keep the pool,
    # class directory and filename together when rebasing that prefix.
    for index, part in enumerate(parts):
        if part in {"train2", "train4", "test2", "test4", "SN"}:
            parts = parts[index:]
            break
    candidates = [base_dir.joinpath(*parts)]
    if fallback_pool:
        candidates.append(base_dir / fallback_pool / original.name)
    return next((path for path in candidates if path.is_file()), None)


def encode_band_scalar(df: pd.DataFrame, path: Path) -> int:
    for column in ("filtercode", "filter", "band"):
        if column in df.columns and df[column].notna().any():
            value = str(df[column].dropna().iloc[0]).lower().strip()
            for key, code in FILTERCODE_MAP.items():
                if key in value:
                    return code
    if "fid" in df.columns and df["fid"].notna().any():
        fid = int(pd.to_numeric(df["fid"].iloc[0], errors="coerce"))
        if fid in (1, 2, 3):
            return fid
    for key, code in FILTERCODE_MAP.items():
        if f"_{key}" in path.name.lower():
            return code
    return 4
