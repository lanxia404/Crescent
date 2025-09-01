# crescent/train/train.py
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crescent.core.config import CrescentCfg
from crescent.core.model import ByteLM

from .utils import choose_device, save_checkpoint, select_dtype, set_seed


def train(config_path: str, data_file: str, out_slot: str):
    cfg = CrescentCfg.load(config_path)
    device = choose_device()
    dtype = select_dtype(cfg.runtime.dtype, device)
    set_seed(cfg.train.seed)

    # === 新增：根據語料長度動態決定有效序列長度，避免 0 樣本 ===
    # 以檔案大小當作 byte 數（對 byte-level 語料等價）
    try:
        corpus_bytes = os.path.getsize(data_file)
    except OSError:
        corpus_bytes = 0
    # 至少要能組出一對 (x,y) → 需要 N >= 2
    if corpus_bytes < 2:
        raise ValueError(
            f"Training data too small: {data_file} has {corpus_bytes} bytes. "
            "Please provide more data (>= 2 bytes)."
        )
    # 有效序列長度：不能超過 N-1，且給個下限避免太小（這裡用 8 你可改）
    seq_len_eff = min(cfg.model.max_seq_len, max(8, corpus_bytes - 1))

    from crescent.train.dataset import ByteDataset  # 確保引用正確

    ds = ByteDataset(data_file, seq_len=seq_len_eff)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

    # ===== 下面保持原樣 =====
    model = ByteLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    model.train()
    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch} (seq_len_eff={seq_len_eff})")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % cfg.train.eval_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 儲存到 slots/<out_slot>/weights/latest.pt
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(root_dir, "slots", out_slot, "weights")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "latest.pt")
    save_checkpoint(
        ckpt_path,
        {
            "model_state": model.state_dict(),
            "config_path": config_path,
        },
    )
    print(f"[Crescent] saved checkpoint => {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_slot", default="B")
    args = ap.parse_args()
    train(args.config, args.data_file, args.out_slot)
