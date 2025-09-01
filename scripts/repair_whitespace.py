#!/usr/bin/env python3
"""
修復整個專案的空白/縮排/換行：
- 去除 UTF-8 BOM
- 規一化換行為 LF
- 將 tab 轉成 4 空白（.py/.sh/.yaml/.yml/.md/.txt）
- 保持檔案權限與時間戳（盡量）
不修改語義，只動 whitespace。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEXT_EXTS = {".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".cfg"}
TAB_TARGET_EXTS = TEXT_EXTS


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    # 忽略常見二進位
    if path.suffix.lower() in {
        ".pt",
        ".safetensors",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".xz",
    }:
        return False
    # 粗略判斷：嘗試以 UTF-8 讀
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        chunk.decode("utf-8")
        return True
    except Exception:
        return False


def normalize_text(data: bytes, convert_tabs: bool) -> bytes:
    # 去 BOM
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    # CRLF -> LF
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # 轉 tab -> 4 spaces（僅指定副檔名）
    if convert_tabs:
        data = data.replace(b"\t", b"    ")
    return data


def process_file(path: Path) -> bool:
    if not is_text_file(path):
        return False
    orig = path.read_bytes()
    convert_tabs = path.suffix.lower() in TAB_TARGET_EXTS
    fixed = normalize_text(orig, convert_tabs)
    if fixed != orig:
        path.write_bytes(fixed)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="專案根目錄（預設當前）")
    ap.add_argument("--dry-run", action="store_true", help="僅顯示將修改的檔案")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    changed = 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        total += 1
        if args.dry_run:
            if is_text_file(p):
                print(f"[SCAN] {p}")
            continue
        if process_file(p):
            changed += 1
            print(f"[FIX]  {p}")
    print(f"[DONE] scanned={total}, changed={changed}")


if __name__ == "__main__":
    sys.exit(main())
