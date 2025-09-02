# crescent/train/utils.py
import os
import random
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch

# 使用統一的位元組↔UTF-8 安全轉換工具
from crescent.core.bytes_codec import bytes_to_utf8_safe


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
    # bf16：優先選，若框架/硬體不支援則回退
    if prefer == "bf16" and getattr(torch, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    # fp16：僅在加速裝置上使用
    if prefer == "fp16" and device.type in {"cuda", "xpu", "mps"}:
        return torch.float16
    # 預設 fp32
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
    """
    將文字以 UTF-8 編碼為 0..255 的位元組 token。
    """
    b = text.encode("utf-8", errors="replace")
    return torch.tensor(list(b), dtype=torch.long)


def _to_byte_list(tokens: Union[torch.Tensor, Sequence[int], bytes, bytearray]) -> List[int]:
    """
    將各種型別的 token 容器轉為 0..255 的位元組 list。
    """
    if isinstance(tokens, torch.Tensor):
        return [int(x) & 0xFF for x in tokens.detach().cpu().reshape(-1).tolist()]
    if isinstance(tokens, (bytes, bytearray)):
        return [int(x) for x in tokens]
    # 假定為整數序列
    return [int(x) & 0xFF for x in tokens]  # type: ignore[arg-type]


def decode_bytes(tokens: Union[torch.Tensor, Sequence[int], bytes, bytearray]) -> str:
    """
    UTF-8 安全解碼：
    - 使用 'replace' 解碼避免例外
    - NFKC 正規化
    - 過濾 C0/C1 控制字元（保留 \t/\n/\r）
    回傳：文字（不含有效度）
    """
    ids = _to_byte_list(tokens)
    s, _ = bytes_to_utf8_safe(bytes(ids))
    return s


def decode_bytes_with_ratio(
    tokens: Union[torch.Tensor, Sequence[int], bytes, bytearray],
) -> Tuple[str, float]:
    """
    與 decode_bytes 相同，但額外回傳 utf8_ok（0.0~1.0，非 U+FFFD 的比例）。
    """
    ids = _to_byte_list(tokens)
    return bytes_to_utf8_safe(bytes(ids))
