#!/usr/bin/env bash
set -euo pipefail

SLOT="${1:-current}"
PORT="${2:-8000}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"
export CRESCENT_SLOT="$SLOT"

# 裝置回報（日誌用途）
python "$ROOT_DIR/scripts/detect_device.py" || true

# 啟動服務
exec uvicorn crescent.runtime.serve:app \
  --host 0.0.0.0 --port "$PORT" --log-level info
