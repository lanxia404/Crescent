import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from crescent.bitnet.model_bitnet import ByteBitNetLM
from crescent.core.config import CrescentCfg
from crescent.train.utils import choose_device, decode_bytes, encode_bytes, select_dtype

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CFG = os.path.join(ROOT_DIR, "configs", "model", "bitnet-b1p58-char-256.yaml")

app = FastAPI(title="Crescent BitNet Runtime")


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    slot: Optional[str] = None


class GenResp(BaseModel):
    slot: str
    device: str
    output: str
    variant: str


_cached = {"slot": None, "model": None, "cfg": None, "device": None, "dtype": None, "variant": None}


def _slot_path(slot: str) -> str:
    if slot == "current":
        return os.path.join(ROOT_DIR, "slots", "current")
    return os.path.join(ROOT_DIR, "slots", slot)


def _load_model(slot: str):
    cfg = CrescentCfg.load(DEFAULT_CFG)
    device = choose_device()
    dtype = select_dtype(cfg.runtime.dtype, device)

    variant = getattr(cfg.model, "bitnet_variant", "1b")
    ckpt = os.path.join(_slot_path(slot), "weights", f"latest_bitnet_{variant}.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = ByteBitNetLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        variant=variant,
        act_bits=getattr(cfg.model, "act_bits", 8),
        group_size=getattr(cfg.model, "group_size", 64),
        use_rope=getattr(cfg.model, "use_rope", False),
    )
    sd = torch.load(ckpt, map_location="cpu")["model_state"]
    model.load_state_dict(sd, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    _cached.update(
        {
            "slot": slot,
            "model": model,
            "cfg": cfg,
            "device": device,
            "dtype": dtype,
            "variant": variant,
        }
    )


def ensure_loaded(slot: str):
    if _cached["model"] is None or _cached["slot"] != slot:
        _load_model(slot)


@app.get("/healthz")
async def healthz():
    slot = os.environ.get("CRESCENT_SLOT", "current")
    try:
        ensure_loaded(slot)
        return {
            "ok": True,
            "slot": slot,
            "device": str(_cached["device"]),
            "variant": _cached["variant"],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "slot": slot}


@app.post("/generate", response_model=GenResp)
async def generate(req: GenReq):
    slot = req.slot or os.environ.get("CRESCENT_SLOT", "current")
    ensure_loaded(slot)
    model = _cached["model"]
    device = _cached["device"]
    idx = encode_bytes(req.prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k
        )
    text = decode_bytes(out[0])
    return GenResp(slot=slot, device=str(device), output=text, variant=_cached["variant"])


@app.get("/")
async def index():
    return {
        "name": "Crescent Runtime",
        "endpoints": ["/healthz", "/generate"],
        "slot_hint": os.environ.get("CRESCENT_SLOT", "current"),
    }


@app.get("/favicon.ico")
async def favicon():
    # 回傳 204，避免 404 噪音
    from fastapi import Response

    return Response(status_code=204)
