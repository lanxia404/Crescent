# crescent/train/train.py
import argparse
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crescent.core.config import CrescentCfg
from crescent.core.model import ByteLM

from .dataset import ByteDataset
from .utils import choose_device, set_seed


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_autocast(device: torch.device, want_bf16: bool):
    if not want_bf16:
        return nullcontext()

    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)

    if device.type == "xpu":
        try:
            return torch.autocast("xpu", dtype=torch.bfloat16)
        except Exception:
            pass
        try:
            amp_mod = getattr(getattr(torch, "xpu", None), "amp", None)
            if amp_mod and hasattr(amp_mod, "autocast"):
                return amp_mod.autocast(dtype=torch.bfloat16)
        except Exception:
            pass
        # 訓練端採靜默回退即可（你也可 print 一條提示）
        return nullcontext()

    if device.type == "mps":
        try:
            return torch.autocast("mps", dtype=torch.bfloat16)
        except Exception:
            pass

    return nullcontext()


def train(config_path: str, data_file: str, out_slot: str):
    cfg = CrescentCfg.load(config_path)
    device = choose_device()
    want_bf16 = str(getattr(cfg.runtime, "dtype", "")).lower() == "bf16"
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
    n_samples = max(1, len(ds))  # 至少 1
    pin = device.type in {"cuda", "xpu"}
    eff_bs = max(1, min(cfg.train.batch_size, n_samples))  # 不超過樣本數，至少 1
    dl = DataLoader(
        ds,
        batch_size=eff_bs,
        shuffle=True,
        drop_last=False,  # 關閉，避免小資料整批掉光
        pin_memory=pin,
    )

    # 建模（權重 fp32，上裝置；精度用 autocast 控制）
    model = ByteLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    ).to(device=device)

    n_params = _count_params(model)
    print(
        f"[Crescent] Model params: {n_params:,}  (seq_len_eff={seq_len_eff})  "
        f"device={device}  want_bf16={want_bf16}  samples={n_samples}  effective_batch={eff_bs}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # XPU + IPEX 最佳化（可用時）
    if device.type == "xpu" and want_bf16:
        try:
            import intel_extension_for_pytorch as ipex

            model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16, inplace=True)
            print("[Crescent] IPEX optimize enabled (dtype=bf16)")
        except Exception as e:
            print(f"[Crescent] IPEX optimize skipped: {e}")

    model.train()
    global_step = 0
    accum = max(1, getattr(cfg.train, "accumulation_steps", 1))
    amp_ctx = _make_autocast(device, want_bf16)

    for epoch in range(1, cfg.train.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
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
    torch.save({"model_state": model.state_dict(), "config_path": config_path}, ckpt_path)
    print(f"[Crescent] saved checkpoint => {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_slot", default="B")
    args = ap.parse_args()
    train(args.config, args.data_file, args.out_slot)
