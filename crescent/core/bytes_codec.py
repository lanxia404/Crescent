# crescent/core/bytes_codec.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import unicodedata
from typing import Iterable, List, Tuple

ALLOW_CTRL = {9, 10, 13}  # \t \n \r


def _filter_control_chars(s: str) -> str:
    out = []
    for ch in s:
        oc = ord(ch)
        if oc < 32 and oc not in ALLOW_CTRL:
            out.append(" ")
        elif 0x7F <= oc <= 0x9F:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def ids_to_bytes(ids: Iterable[int]) -> bytes:
    return bytes((int(t) & 0xFF for t in ids))


def bytes_to_utf8_safe(b: bytes) -> Tuple[str, float]:
    """
    以 'replace' 解碼，並以「非替代字元比例」估計 UTF-8 有效度，範圍 0.0~1.0。
    """
    if not b:
        return "", 1.0
    s = b.decode("utf-8", errors="replace")
    total = len(s)
    # U+FFFD = '�'（替代字元）；以其占比估算無效度
    repl = s.count("\ufffd")
    valid_ratio = 1.0 if total == 0 else (total - repl) / total
    # 正規化與控制字元過濾
    s = unicodedata.normalize("NFKC", s)
    s = _filter_control_chars(s)
    # 防禦性夾入 [0,1]
    if valid_ratio < 0.0:
        valid_ratio = 0.0
    if valid_ratio > 1.0:
        valid_ratio = 1.0
    return s, valid_ratio


def detok_utf8_safe(ids: List[int]) -> Tuple[str, float]:
    return bytes_to_utf8_safe(ids_to_bytes(ids))
