# Crescent

一個最小但可用的 LLM 專案骨架，從資料→訓練→部署一路打通，支援 **A/B 分槽原子切換** 與 **多裝置推理**（**Intel XPU** / CUDA / CPU）。  
預設路線為 **byte-level（vocab=256）decoder-only Transformer**，適合用來學習、實驗與擴充。

---

## ✨ 近期重點更新

- **Crawler 修復**：改用 `tldextract.top_domain_under_public_suffix`、支援 `--allow-registered`（含子網域）、`robots.txt`/`sitemap`、自訂 UA、**即時日誌**、輸出 **JSONL 分片**。  
- **Corpus 建構**：`build_corpus.py` 支援 **HTML/JSONL/JSON/txt**，**移除頁面標題**、**全域行級＋文件級去重**、Unicode 正規化、即時日誌。  
- **推理輸出修正**：加入 **UTF-8 安全解碼** 與控制字元過濾，API 回傳 `utf8_ok` 品質指標，避免亂碼。  
- **生成策略**：模型 `generate()` 支援 **top-k、top-p（nucleus）與重複抑制（repetition_penalty）**。  
- **BF16 支援**：訓練與推理採 **autocast(bfloat16)**（權重維持 FP32，更穩定），XPU 上自動嘗試 IPEX 最佳化；XPU 未支援 autocast 時**優雅回退** FP32。  
- **Dataset 健壯化**：`ByteDataset` 採滑動窗切片，**極小語料也保證 ≥1 樣本**，避免 `epoch 0it`。  
- **服務日誌**：所有關鍵腳本均為**即時 flush** 日誌，便於觀測。

---

## 功能總覽

- **從零開始訓練**：極簡 byte-level Transformer；支援梯度累積、（可選）BF16 自動混合精度。  
- **資料→語料**：網站抓取（尊重 robots）、正文抽取、去標題、全域去重。  
- **服務化推理**：FastAPI `GET /healthz`、`POST /generate`（帶抽樣與重複抑制）。  
- **A/B 分槽**：在 `B` 槽通過健康檢查後，`promote.sh` 原子切到 `current`。  
- **裝置自動偵測**：優先 **Intel XPU**（需要 PyTorch XPU，選配 IPEX），再來 CUDA，最後 CPU。

---

## 安裝

```bash
# 建議使用虛擬環境
python -m venv .venv && source .venv/bin/activate

# 通用依賴（CPU / CUDA / XPU 皆可安裝）
pip install -r requirements.txt

# 若使用 Intel XPU（可選）
# 依你的平台／驅動版本安裝對應的 PyTorch-XPU + IPEX（版本需匹配）
pip install -r requirements-intel-xpu.txt || true
```

> ⚠️ 若 IPEX 與 PyTorch-XPU 版本不匹配，會看到類似 `module 'torch.xpu' has no attribute '...'` 的訊息。專案已自動**跳過最佳化並回退**，不影響功能。

---

## 目錄結構（節選）

```
Crescent/
├─ configs/
│  └─ model/
│     └─ tiny-char-256.yaml        # 模型＋訓練＋runtime 設定（支援 runtime.dtype: bf16）
├─ crescent/
│  ├─ core/
│  │  ├─ model.py                  # ByteLM（含 generate: top-k/top-p/repetition_penalty）
│  │  └─ bytes_codec.py            # UTF-8 安全解碼與控制字元過濾
│  ├─ runtime/
│  │  └─ serve.py                  # FastAPI（UTF-8 safe、bf16 autocast 容錯、即時日誌）
│  └─ train/
│     ├─ dataset.py                # ByteDataset（滑動窗，極小語料也能取樣）
│     ├─ train.py                  # 訓練（bf16 autocast；XPU 嘗試 IPEX）
│     └─ utils.py                  # 裝置選擇、編解碼、安全存檔等
├─ scripts/
│  ├─ crawl_site.py                # Crawler（robots/sitemap/UA、JSONL 分片、即時日誌）
│  ├─ build_corpus.py              # 語料建構（去標題/去重/正規化、即時日誌）
│  ├─ train_from_scratch.sh        # 範例訓練流程（輸出到 slots/B）
│  ├─ run_slot.sh                  # 啟動某槽位服務（如 current:8000）
│  ├─ promote.sh                   # 將 A/B 切到 current（原子切換）
│  └─ format_python.sh             # 格式化＋基礎靜態檢查
└─ slots/
   ├─ A/weights/latest.pt
   ├─ B/weights/latest.pt
   └─ current -> (A 或 B)
```

---

## 快速開始

### 1) 取得資料 → 建語料

```bash
# 抓取（以中文維基起手，可換任意站；不限語言）
python scripts/crawl_site.py \
  --seeds https://zh.wikipedia.org \
  --out-dir data/crawl/wiki \
  --use-sitemap \
  --max-pages 2000 \
  --lang zh \
  --allow-registered \
  --delay 0.8

# 由原始抓檔建「乾淨語料」
python scripts/build_corpus.py \
  --in-path data/crawl/wiki \
  --out-file data/wiki_corpus.clean.txt \
  --min-line-chars 18 \
  --log-interval 200
```

> `build_corpus.py` 會：抽正文、移除頁面標題（`<title>` / `og:title` / `h1` 等）、行級＋文件級去重、Unicode 正規化。
> 亦可直接針對 `.txt` 做行級去重（但無法精準去標題）。

### 2) 訓練

`configs/model/tiny-char-256.yaml`（節選）：

```yaml
model:
  vocab_size: 256
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  dropout: 0.1
  max_seq_len: 1024
# bitnet_* 欄位保留為 null 即可（預留未來 1/1.58 低比特路線）

train:
  batch_size: 32
  lr: 3.0e-4
  weight_decay: 0.01
  epochs: 3
  eval_interval: 200
  seed: 42
  accumulation_steps: 4

runtime:
  dtype: bf16  # 支援裝置時將以 autocast(bf16) 執行；否則自動回退 fp32
```

啟動訓練（預設輸出到 `slots/B`）：

```bash
bash scripts/train_from_scratch.sh
```

> 訓練端重點：
>
> * 權重維持 FP32，上下文以 **autocast(bf16)** 計算（XPU 未支援時自動回 FP32）。
> * **極小資料也能訓練**：`ByteDataset` 滑動窗切片 + `drop_last=False` + 自適應 batch。
> * 會印出參數量、樣本數、effective batch、裝置與 bf16 狀態。

### 3) 啟動服務與切換

```bash
# 啟動某槽位服務
bash scripts/run_slot.sh B 8001

# 健康檢查通過後，將 current 切到 B（原子切換）
bash scripts/promote.sh B 8001

# 之後用 current（預設 8000）
bash scripts/run_slot.sh current 8000
```

---

## 推理 API

### `GET /healthz`

回傳：

```json
{"ok": true, "slot": "current", "device": "xpu", "dtype": "torch.float32"}
```

### `POST /generate`

**Request**

```json
{
  "prompt": "Hello",
  "max_new_tokens": 64,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.2,
  "slot": "current"
}
```

**Response**

```json
{
  "slot": "current",
  "device": "xpu",
  "output": "Hello ...",
  "utf8_ok": 0.997
}
```

* `utf8_ok`：新生成位元組經 UTF-8 解碼的**有效比例**（0\~1），可用來快速評估亂碼風險。
* 服務端僅將「**新產生**」的 token 轉文字（不重覆包含 prompt）。
* XPU 未支援 autocast 時自動回退 FP32，並在日誌提示。

---

## 常見疑難排解

* **推理亂碼 / 充滿 `�`**：已改為 UTF-8 安全解碼；若 `utf8_ok` 仍低，通常是模型未收斂或載入到錯的 checkpoint。
* **輸出 `the the the…` 重複**：開啟 `top_p/top_k` 與 `repetition_penalty`（建議 1.1\~1.3），或加大訓練步數與語料。
* **訓練 `epoch 0it`**：語料過小＋`drop_last=True` 會整批丟掉；本專案已改用滑動窗切片、`drop_last=False` 並自適應 batch。
* **XPU autocast 錯誤**：若見 `torch.xpu.amp.autocast` 不存在，代表版本組合不支援；已自動回 FP32。建議升級到匹配的 PyTorch-XPU/IPEX。
* **Crawler 0 頁**：確保使用 `--allow-registered` 放寬子網域、設置 UA、開 `--use-sitemap`，並檢查 `robots.txt` 與日誌。
* **格式／靜態檢查**：`bash scripts/format_python.sh`。

---

## 安全與合規

* Crawler **預設尊重 `robots.txt`**；請確保資料來源符合授權與使用條款。
* 語料清理腳本會做標題過濾與去重，但不保證百分百過濾所有受保護內容；用於公開場景前需額外稽核。

---

## 發展路線（草案）

* Dense 與 **BitNet(1/1.58)** 對照路線（目前保留設定鍵，後續實作前向偽量化原型）。
* 最小評測：held-out perplexity、簡單基準。
* 更完整推理服務：流式輸出、監控（Prometheus/Grafana）與 A/B 指標面板。
* 子詞（BPE/SentencePiece）路線與中文優化。

---

## 貢獻

歡迎 Issue/PR。請在提交前執行：

```bash
bash scripts/format_python.sh
```

---
