#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

# 預設訓練輸出到 B 槽
OUT_SLOT="${1:-B}"
CONFIG_PATH="$ROOT_DIR/configs/model/tiny-char-256.yaml"

mkdir -p "$ROOT_DIR/slots/$OUT_SLOT/weights" "$ROOT_DIR/slots/$OUT_SLOT/runtime"

python -m crescent.train.train \
  --config "$CONFIG_PATH" \
  --out_slot "$OUT_SLOT" \
  --data_file "$ROOT_DIR/data/sample.txt"
