#!/usr/bin/env python3
"""Submit a PDF to MinerU batch API and download the parsed result zip.

Token sources, in order:
1. MINERU_API_TOKEN environment variable
2. hidden TTY prompt
3. stdin

Never pass the token as a command-line argument.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


API_BASE = "https://mineru.net/api/v4"


def read_token() -> str:
    token = os.environ.get("MINERU_API_TOKEN", "").strip()
    if token:
        return token
    print("No MINERU_API_TOKEN found. Apply for a token at https://mineru.net/apiManage/token", file=sys.stderr)
    if sys.stdin.isatty():
        return getpass.getpass("MinerU API token: ").strip()
    return sys.stdin.read().strip()


def make_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def infer_object_name(file_path: Path) -> str:
    return file_path.name.replace(" ", "_")


def create_batch(token: str, file_path: Path, model_version: str, data_id: str | None) -> dict:
    payload = {
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
        "files": [
            {
                "name": infer_object_name(file_path),
                "is_ocr": True,
                "data_id": data_id or file_path.stem,
            }
        ],
        "model_version": model_version,
    }
    resp = requests.post(f"{API_BASE}/file-urls/batch", headers=make_headers(token), data=json.dumps(payload), timeout=60)
    print("create_status", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") not in (0, "0", None):
        raise RuntimeError(f"create batch failed: {data}")
    return data["data"]


def upload_file(url: str, file_path: Path) -> None:
    with file_path.open("rb") as fh:
        resp = requests.put(url, data=fh, timeout=600)
    print("upload_status", resp.status_code)
    resp.raise_for_status()


def poll_result(token: str, batch_id: str, timeout_s: int, interval_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        resp = requests.get(f"{API_BASE}/extract-results/batch/{batch_id}", headers=make_headers(token), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") not in (0, "0", None):
            raise RuntimeError(f"poll failed: {data}")
        extract_result = data["data"]["extract_result"]
        state = extract_result[0].get("state") if extract_result else data["data"].get("state")
        if state != last_state:
            print("state", state)
            last_state = state
        if state == "done":
            return extract_result[0]
        if state == "failed":
            raise RuntimeError(f"MinerU extraction failed: {extract_result[0]}")
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for batch {batch_id}")


def safe_zip_name(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def download_and_extract(result: dict, out_dir: Path, fallback_name: str) -> Path:
    zip_url = result.get("full_zip_url")
    if not zip_url:
        raise RuntimeError(f"full_zip_url missing from result: {result}")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / safe_zip_name(zip_url, f"{fallback_name}.zip")
    with requests.get(zip_url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with zip_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    extract_dir = out_dir / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print("zip_path", zip_path)
    print("extract_dir", extract_dir)
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MinerU batch extraction for a PDF.")
    parser.add_argument("--file", required=True, help="PDF file path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model-version", default="vlm", choices=["vlm", "pipeline"], help="MinerU model version")
    parser.add_argument("--data-id", default=None, help="Optional MinerU data_id")
    parser.add_argument("--timeout", type=int, default=1800, help="Polling timeout in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    token = read_token()
    if not token:
        raise RuntimeError("MinerU API token is empty")

    batch = create_batch(token, file_path, args.model_version, args.data_id)
    batch_id = batch["batch_id"]
    print("batch_id", batch_id)
    upload_url = batch["file_urls"][0]
    upload_file(upload_url, file_path)
    result = poll_result(token, batch_id, args.timeout, args.interval)
    extract_dir = download_and_extract(result, Path(args.out), file_path.stem)
    for path in sorted(extract_dir.rglob("*")):
        if path.is_file():
            print("file", path.relative_to(extract_dir))


if __name__ == "__main__":
    main()
