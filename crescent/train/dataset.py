import torch
from torch.utils.data import Dataset


class ByteDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        with open(path, "rb") as f:
            self.data = f.read()
        self.seq_len = seq_len
        self.bytes = torch.tensor(list(self.data), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.bytes) - self.seq_len)

    def __getitem__(self, i):
        x = self.bytes[i : i + self.seq_len]
        y = self.bytes[i + 1 : i + self.seq_len + 1]
        return x, y
