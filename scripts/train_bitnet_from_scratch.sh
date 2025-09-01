#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

CFG="${1:-$ROOT_DIR/configs/model/bitnet-b1p58-char-256.yaml}"
OUT_SLOT="${2:-B}"
DATA="${3:-$ROOT_DIR/data/sample.txt}"

mkdir -p "$ROOT_DIR/slots/$OUT_SLOT/weights" "$ROOT_DIR/slots/$OUT_SLOT/runtime"

python -m crescent.train.train_bitnet \
  --config "$CFG" \
  --data_file "$DATA" \
  --out_slot "$OUT_SLOT"
