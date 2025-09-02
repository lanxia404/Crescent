# crescent/train/dataset.py
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class ByteDataset(Dataset):
    """
    將整個語料以 byte-level 切成 (x, y)：
      x: 長度 seq_len
      y: x 的下一個位元組（右移一位）
    使用滑動窗口，預設 stride = max(1, seq_len // 2)。
    """

    def __init__(self, path: str, seq_len: int, stride: int | None = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            buf = f.read()
        if len(buf) < 2:
            raise ValueError(f"data too small: {path} ({len(buf)} bytes)")

        self.data = torch.tensor(list(buf), dtype=torch.long)
        self.seq_len = int(seq_len)
        self.n = int(self.data.numel())
        self.max_start = max(0, self.n - self.seq_len - 1)
        self.stride = int(stride) if stride is not None else max(1, self.seq_len // 2)

        # 至少回報 1 個樣本，避免 DataLoader 沒 batch
        self._len = 1 if self.max_start == 0 else 1 + (self.max_start // self.stride)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if self.max_start <= 0:
            start = 0
        else:
            # 緩解越界：最後一個樣本貼齊資料尾端
            start = min(int(idx) * self.stride, self.max_start)

        end = start + self.seq_len
        x = self.data[start:end]  # [seq_len]
        y = self.data[start + 1 : end + 1]  # [seq_len]
        return x, y
