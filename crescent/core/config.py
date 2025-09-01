import os
from dataclasses import dataclass
from typing import Literal, Optional

import yaml

DType = Literal["bf16", "fp16", "fp32"]


@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    max_seq_len: int
    # BitNet（可選）
    bitnet_variant: Optional[str] = None
    act_bits: Optional[int] = None
    group_size: Optional[int] = None
    use_rope: Optional[bool] = None


@dataclass
class TrainCfg:
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    eval_interval: int
    seed: int
    accumulation_steps: int = 1  # 新增


@dataclass
class RuntimeCfg:
    dtype: DType


@dataclass
class CrescentCfg:
    model: ModelCfg
    train: TrainCfg
    runtime: RuntimeCfg

    @staticmethod
    def load(path: str) -> "CrescentCfg":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        m = ModelCfg(**raw["model"])
        t = TrainCfg(**raw["train"])
        r = RuntimeCfg(**raw["runtime"])
        return CrescentCfg(model=m, train=t, runtime=r)
