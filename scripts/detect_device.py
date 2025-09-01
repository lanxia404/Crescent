#!/usr/bin/env python
import json

import torch


def _xpu_available():
    try:
        pass  # 觸發 XPU 後端註冊
    except Exception:
        pass
    has_xpu_attr = hasattr(torch, "xpu")
    is_avail = False
    if has_xpu_attr:
        is_avail = getattr(torch.xpu, "is_available", lambda: False)()
    return has_xpu_attr and is_avail


def detect_device():
    if _xpu_available():
        return {"device": "xpu"}

    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
        }

    # Apple MPS（可選）
    try:
        from torch.backends import mps

        if mps.is_available():
            return {"device": "mps"}
    except Exception:
        pass

    return {"device": "cpu"}


if __name__ == "__main__":
    print("[Crescent] device:", json.dumps(detect_device(), ensure_ascii=False))
