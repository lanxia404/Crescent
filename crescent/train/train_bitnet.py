import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crescent.bitnet.model_bitnet import ByteBitNetLM
from crescent.core.config import CrescentCfg
from crescent.train.dataset import ByteDataset
from crescent.train.utils import choose_device, save_checkpoint, select_dtype, set_seed


def train(config_path: str, data_file: str, out_slot: str):
    cfg = CrescentCfg.load(config_path)
    device = choose_device()
    dtype = select_dtype(cfg.runtime.dtype, device)
    set_seed(cfg.train.seed)

    ds = ByteDataset(data_file, seq_len=cfg.model.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

    variant = getattr(cfg.model, "bitnet_variant", "1b")
    act_bits = getattr(cfg.model, "act_bits", 8)
    group_size = getattr(cfg.model, "group_size", 64)
    use_rope = getattr(cfg.model, "use_rope", False)

    model = ByteBitNetLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        variant=variant,
        act_bits=act_bits,
        group_size=group_size,
        use_rope=use_rope,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    model.train()
    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        pbar = tqdm(dl, desc=f"[BitNet:{variant}] epoch {epoch}")
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

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "slots", out_slot, "weights"
    )
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"latest_bitnet_{variant}.pt")
    save_checkpoint(ckpt_path, {"model_state": model.state_dict(), "config_path": config_path})
    print(f"[Crescent] saved BitNet checkpoint => {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_slot", default="B")
    args = ap.parse_args()
    train(args.config, args.data_file, args.out_slot)
