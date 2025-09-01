import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crescent.core.config import CrescentCfg
from crescent.core.model import ByteLM

from .dataset import ByteDataset
from .utils import choose_device, save_checkpoint, select_dtype, set_seed


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(config_path: str, data_file: str, out_slot: str):
    cfg = CrescentCfg.load(config_path)
    device = choose_device()
    dtype = select_dtype(cfg.runtime.dtype, device)
    set_seed(cfg.train.seed)

    # 自適應序列長度（避免 0 樣本）
    try:
        corpus_bytes = os.path.getsize(data_file)
    except OSError:
        corpus_bytes = 0
    if corpus_bytes < 2:
        raise ValueError(f"Training data too small: {data_file} = {corpus_bytes} bytes")
    seq_len_eff = min(cfg.model.max_seq_len, max(8, corpus_bytes - 1))

    ds = ByteDataset(data_file, seq_len=seq_len_eff)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

    model = ByteLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    ).to(device=device, dtype=dtype)

    n_params = _count_params(model)
    print(
        f"[Crescent] Model params: {
            n_params:,}  (seq_len_eff={seq_len_eff})"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    model.train()
    global_step = 0
    accum = max(1, getattr(cfg.train, "accumulation_steps", 1))
    # 可視需求啟 AMP

    for epoch in range(1, cfg.train.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            (loss / accum).backward()

            if i % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg.train.eval_interval == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "slots", out_slot, "weights"
    )
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "latest.pt")
    save_checkpoint(ckpt_path, {"model_state": model.state_dict(), "config_path": config_path})
    print(f"[Crescent] saved checkpoint => {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_slot", default="B")
    args = ap.parse_args()
    train(args.config, args.data_file, args.out_slot)
