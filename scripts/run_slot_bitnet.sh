#!/usr/bin/env bash
set -euo pipefail
SLOT="${1:-current}"
PORT="${2:-8002}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"
export CRESCENT_SLOT="$SLOT"

python "$ROOT_DIR/scripts/detect_device.py" || true
exec uvicorn crescent.runtime.serve_bitnet:app --host 0.0.0.0 --port "$PORT" --log-level info
