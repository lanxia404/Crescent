# crescent/core/config.py
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
    # BitNet 選用欄位
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

        def _require_map(section: str):
            val = None if raw is None else raw.get(section)
            if not isinstance(val, dict):
                raise ValueError(
                    f"YAML section '{section}' is missing or not a mapping in {path}.\n"
                    f"Got: {type(val).__name__ if val is not None else 'None'}\n"
                    f"Make sure your YAML looks like:\n"
                    f'"""\n{section}:\n  key: value\n"""\n'
                )
            return val

        m = ModelCfg(**_require_map("model"))
        t = TrainCfg(**_require_map("train"))
        r = RuntimeCfg(**_require_map("runtime"))
        return CrescentCfg(model=m, train=t, runtime=r)
