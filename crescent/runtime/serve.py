# crescent/runtime/serve.py
import logging
import os
from contextlib import nullcontext
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from crescent.core.bytes_codec import detok_utf8_safe  # 將 byte-ids 安全轉 UTF-8
from crescent.core.config import CrescentCfg
from crescent.core.model import ByteLM
from crescent.train.utils import choose_device, encode_bytes, select_dtype

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CFG = os.path.join(ROOT_DIR, "configs", "model", "train.yaml")

app = FastAPI(title="Crescent Runtime")


# ---- Logging（即時 flush）----
class LineFlushHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg + "\n")
        try:
            stream.flush()
        except Exception:
            pass


_logger = logging.getLogger("crescent.runtime")
if not _logger.handlers:
    _h = LineFlushHandler()
    _h.setFormatter(logging.Formatter("[Crescent][serve] %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None  # 新增
    repetition_penalty: Optional[float] = None  # 新增（建議 1.1~1.3）
    slot: Optional[str] = None


class GenResp(BaseModel):
    slot: str
    device: str
    output: str
    utf8_ok: float  # 0.0~1.0


_cached = {"slot": None, "model": None, "cfg": None, "device": None, "dtype": None}


def _slot_path(slot: str) -> str:
    if slot == "current":
        return os.path.join(ROOT_DIR, "slots", "current")
    return os.path.join(ROOT_DIR, "slots", slot)


def _load_model(slot: str):
    cfg = CrescentCfg.load(DEFAULT_CFG)
    device = choose_device()
    dtype = select_dtype(cfg.runtime.dtype, device)

    ckpt = os.path.join(_slot_path(slot), "weights", "latest.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = ByteLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    )
    sd = torch.load(ckpt, map_location="cpu")["model_state"]
    model.load_state_dict(sd, strict=True)
    model.to(device=device)  # 權重維持 fp32；精度用 autocast 控制
    model.eval()

    _cached.update({"slot": slot, "model": model, "cfg": cfg, "device": device, "dtype": dtype})

    # 啟動日誌
    try:
        import time

        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt)))
    except Exception:
        mtime = "unknown"
    params = sum(p.numel() for p in model.parameters())
    _logger.info(
        f"loaded slot={slot} device={device} dtype={dtype} ckpt={ckpt} mtime={mtime} params={params:_}"
    )


def ensure_loaded(slot: str):
    if _cached["model"] is None or _cached["slot"] != slot:
        _load_model(slot)
        _cached["slot"] = slot


def _amp_context_for(device: torch.device):
    """Return a best-effort autocast(bf16) context for the given device."""
    # CUDA
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)

    # XPU: 依序嘗試 torch.autocast("xpu") -> torch.xpu.amp.autocast -> 退回 FP32
    if device.type == "xpu":
        # 1) torch.autocast("xpu", ...)
        try:
            return torch.autocast("xpu", dtype=torch.bfloat16)
        except Exception:
            pass
        # 2) torch.xpu.amp.autocast(...)
        try:
            amp_mod = getattr(getattr(torch, "xpu", None), "amp", None)
            if amp_mod and hasattr(amp_mod, "autocast"):
                return amp_mod.autocast(dtype=torch.bfloat16)
        except Exception:
            pass
        # 3) 無法使用 autocast：回 FP32 並告警一次
        _logger.warning(
            "XPU bf16 autocast not available; running in fp32. "
            "Consider upgrading matching PyTorch-XPU and IPEX."
        )
        return nullcontext()

    # MPS
    if device.type == "mps":
        try:
            return torch.autocast("mps", dtype=torch.bfloat16)
        except Exception:
            pass

    # 其他裝置：FP32
    return nullcontext()


@app.get("/healthz")
async def healthz():
    slot = os.environ.get("CRESCENT_SLOT", "current")
    try:
        ensure_loaded(slot)
        return {
            "ok": True,
            "slot": slot,
            "device": str(_cached["device"]),
            "dtype": str(_cached["dtype"]),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "slot": slot}


@app.post("/generate", response_model=GenResp)
async def generate(req: GenReq):
    slot = req.slot or os.environ.get("CRESCENT_SLOT", "current")
    ensure_loaded(slot)

    model: ByteLM = _cached["model"]  # type: ignore[assignment]
    device = _cached["device"]

    # 編碼輸入為 byte-ids（0..255）
    idx = encode_bytes(req.prompt).unsqueeze(0).to(device)
    in_len = idx.shape[1]  # 之後要用來切出「新增」tokens

    # 選擇對應裝置的 bf16 autocast
    amp_ctx = _amp_context_for(device)

    # 生成完整序列（通常包含輸入）
    with torch.no_grad():
        with amp_ctx:
            out = model.generate(
                idx,
                max_new_tokens=int(req.max_new_tokens),
                temperature=float(req.temperature),
                top_k=int(req.top_k) if req.top_k is not None else None,
                top_p=float(req.top_p) if req.top_p is not None else None,  # 新增
                repetition_penalty=(
                    float(req.repetition_penalty) if req.repetition_penalty else None
                ),  # 新增
            )
    # 只取「新產生」的部份解碼
    out_ids: List[int] = out[0].tolist()
    new_ids: List[int] = out_ids[in_len:] if len(out_ids) > in_len else []
    text, valid_ratio = detok_utf8_safe(new_ids)

    _logger.info(
        "slot=%s device=%s prompt_len=%d new_tokens=%d utf8_ok=%.2f temp=%.2f top_k=%s",
        slot,
        device,
        in_len,
        len(new_ids),
        valid_ratio,
        req.temperature,
        str(req.top_k),
    )

    return GenResp(slot=slot, device=str(device), output=text, utf8_ok=round(valid_ratio, 3))


@app.get("/")
async def index():
    return {
        "name": "Crescent Runtime",
        "endpoints": ["/healthz", "/generate"],
        "slot_hint": os.environ.get("CRESCENT_SLOT", "current"),
    }


@app.get("/favicon.ico")
async def favicon():
    from fastapi import Response

    return Response(status_code=204)
