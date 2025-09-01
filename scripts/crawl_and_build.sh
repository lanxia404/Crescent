#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/crawl}"
LANG="${LANG:-}" # 例如 zh-cn / zh-tw / en
MAXP="${MAXP:-1000}"
CONC="${CONC:-8}"
RPS="${RPS:-2.0}"

# 修改成你想抓的站點或起始 URL
SEEDS=(
  "https://www.example.com"
)

python "$ROOT/scripts/crawl_site.py" \
  --seeds "${SEEDS[@]}" \
  --out-dir "$OUT" \
  ${LANG:+--lang "$LANG"} \
  --max-pages "$MAXP" \
  --concurrency "$CONC" \
  --rps "$RPS" \
  --use-sitemap

python "$ROOT/scripts/build_corpus.py" \
  --in-dir "$OUT" \
  --out-file "$ROOT/data/crawl_corpus.txt"

echo "[OK] Ready corpus => $ROOT/data/crawl_corpus.txt"
