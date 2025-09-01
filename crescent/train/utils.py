import os
import random

import numpy as np
import torch


def choose_device():
    # 嘗試載入 IPEX，以確保 XPU 被註冊
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except Exception:
        pass

    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        from torch.backends import mps

        if mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def select_dtype(prefer: str, device: torch.device):
    prefer = (prefer or "").lower()
    if prefer == "bf16" and getattr(torch, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    if prefer == "fp16" and device.type in {"cuda", "xpu", "mps"}:
        return torch.float16
    return torch.float32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")


def encode_bytes(text: str) -> torch.Tensor:
    b = text.encode("utf-8", errors="replace")
    return torch.tensor(list(b), dtype=torch.long)


def decode_bytes(tokens: torch.Tensor) -> str:
    b = bytes([int(x) % 256 for x in tokens.cpu().tolist()])
    return b.decode("utf-8", errors="replace")
