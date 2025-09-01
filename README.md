# Crescent


一個最小但完整的 LLM 專案骨架，支援 **A/B 分槽原子切換** 與 **通用裝置推理**（Intel XPU / CUDA / CPU）。


## 功能
- 從零開始訓練：極簡字元級（byte-level，vocab=256）Decoder-only Transformer。
- 服務化推理：FastAPI `/healthz` 與 `/generate`。
- A/B 分槽：在 `B` 槽通過健康檢查後，用 `promote.sh` 原子切換至 `current`。
- 裝置自動偵測：優先使用 **Intel XPU**（本地 PyTorch XPU），其次 CUDA（NVIDIA/ROCm），最後 CPU。


## 安裝


```bash
# 建議使用虛擬環境
python -m venv .venv && source .venv/bin/activate


# 通用依賴（CPU/NVIDIA/AMD/Intel 通用）
pip install -r requirements.txt


# 若需 Intel XPU（選擇性）
# 取決於你的驅動與 IPEX 版本，請視實際平台調整 requirements-intel-xpu.txt
pip install -r requirements-intel-xpu.txt || true

```

##  快速開始


```bash
# 1) 準備資料（可直接覆蓋 data/sample.txt）
$ sed -n '1,50p' data/sample.txt


# 2) 從零開始訓練（預設輸出到 slots/B）
bash scripts/train_from_scratch.sh


# 3) 啟動 B 槽推理服務（預設 8001）
bash scripts/run_slot.sh B 8001


# 4) 健康檢查通過後，將 current 切到 B（原子切換）
bash scripts/promote.sh B 8001


# 之後可直接啟動 current 槽服務
bash scripts/run_slot.sh current 8000

```

## 生成範例

```bash

curl -s -X POST "http://127.0.0.1:8000/generate" \
-H 'Content-Type: application/json' \
-d '{"prompt":"Hello","max_new_tokens":64,"temperature":0.8}'

```

## 結構說明

* slots/A|B/weights/latest.pt：模型權重與訓練狀態快照。

* slots/current：符號連結，指向線上使用中的槽（A 或 B）。

* configs/model/tiny-char-256.yaml：模型結構與訓練超參數。

> 本專案為最小雛形，方便你後續擴充（RAG、MoE、KV-Cache、進階代碼自演化等）。
