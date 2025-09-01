#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

# 端點（可改）
DENSE_URL="${1:-http://127.0.0.1:8000}"
BITNET_URL="${2:-http://127.0.0.1:8002}"

# 測試參數（可改）
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. 你好，世界！}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-}"

# 輸出到哪個槽（Dense 預設 current；BitNet 預設 B），僅影響報表輸出目錄
DENSE_SLOT="${DENSE_SLOT:-current}"
BITNET_SLOT="${BITNET_SLOT:-B}"

python "$ROOT_DIR/scripts/bench_compare.py" \
  --dense-url "$DENSE_URL" \
  --bitnet-url "$BITNET_URL" \
  --prompt "$PROMPT" \
  --runs "$RUNS" \
  --warmup "$WARMUP" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  ${TOP_K:+--top-k "$TOP_K"} \
  --dense-slot "$DENSE_SLOT" \
  --bitnet-slot "$BITNET_SLOT"
