#!/usr/bin/env bash
set -euo pipefail

NEXT_SLOT="${1:-B}"
PORT="${2:-8000}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SLOTS_DIR="$ROOT_DIR/slots"
CURRENT_LINK="$SLOTS_DIR/current"
HEALTH_URL="http://127.0.0.1:${PORT}/healthz"

if [[ ! -d "$SLOTS_DIR/$NEXT_SLOT" ]]; then
  echo "[ERR] 目標槽不存在: $SLOTS_DIR/$NEXT_SLOT" >&2
  exit 1
fi

# 健康檢查（等待 60s）
TIMEOUT=60
START=$(date +%s)
while true; do
  if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    break
  fi
  NOW=$(date +%s)
  if ((NOW - START > TIMEOUT)); then
    echo "[ERR] 健康檢查逾時：$HEALTH_URL" >&2
    exit 2
  fi
  sleep 1
done

# 原子切換符號連結
ln -sfn "$SLOTS_DIR/$NEXT_SLOT" "$CURRENT_LINK"
echo "[OK] 已將 current 切換至: $NEXT_SLOT"
