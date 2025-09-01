#!/usr/bin/env python3
import argparse
import json
import math
import os
import pathlib
import statistics
import time

import psutil
import requests


def jpost(url, payload):
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def jget(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def approx_tokens(old: str, new: str) -> int:
    # byte-level vocab: 用 bytes 差估 token 數
    a = len(old.encode("utf-8", "replace"))
    b = len(new.encode("utf-8", "replace"))
    return max(b - a, 1)


def bench_endpoint(name, base_url, prompt, runs, warmup, max_new_tokens, temperature, top_k):
    health = jget(f"{base_url}/healthz")
    device = health.get("device", "unknown")
    slot = health.get("slot", "unknown")

    # payload
    body = {"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": float(temperature)}
    if top_k is not None:
        body["top_k"] = int(top_k)

    # warmup
    for _ in range(warmup):
        try:
            jpost(f"{base_url}/generate", body)
        except Exception:
            pass

    lat, tps, outs = [], [], []
    for _ in range(runs):
        t0 = time.time()
        out = jpost(f"{base_url}/generate", body)
        dt = time.time() - t0
        text = out.get("output", "")
        ns = approx_tokens(prompt, text)
        lat.append(dt)
        tps.append(ns / dt)
        outs.append(text)

    stats = {
        "name": name,
        "url": base_url,
        "slot": slot,
        "device": device,
        "runs": runs,
        "warmup": warmup,
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature),
        "top_k": top_k,
        "p50_latency_s": statistics.median(lat),
        "p95_latency_s": percentile(lat, 95),
        "avg_toks_per_s": sum(tps) / len(tps),
        "min_toks_per_s": min(tps),
        "max_toks_per_s": max(tps),
        "rss_mb": rss_mb(),
    }
    return stats, lat, tps, outs


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def rss_mb():
    try:
        p = psutil.Process()
        return int(p.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def dump_report(out_dir, name, stats, lat, tps):
    ensure_dir(out_dir)
    # JSON
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    # CSV（逐次樣本）
    csv = os.path.join(out_dir, f"{name}_runs.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("run,latency_s,tokens_per_s\n")
        for i, (a, b) in enumerate(zip(lat, tps), start=1):
            f.write(f"{i},{a:.6f},{b:.3f}\n")
    return csv


def maybe_plot(out_dir, dense_csv, bitnet_csv):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return
    try:
        d = pd.read_csv(dense_csv)
        b = pd.read_csv(bitnet_csv)
        # latency boxplot
        plt.figure()
        plt.boxplot([d["latency_s"], b["latency_s"]], labels=["Dense", "BitNet"])
        plt.ylabel("Latency (s)")
        plt.title("Latency Distribution")
        plt.savefig(os.path.join(out_dir, "latency_boxplot.png"), bbox_inches="tight")
        # toks/s boxplot
        plt.figure()
        plt.boxplot([d["tokens_per_s"], b["tokens_per_s"]], labels=["Dense", "BitNet"])
        plt.ylabel("Tokens per Second")
        plt.title("Throughput Distribution")
        plt.savefig(os.path.join(out_dir, "throughput_boxplot.png"), bbox_inches="tight")
    except Exception:
        pass


def calc_ppl_dense(cfg_path, ckpt_path, sample_path):
    import torch

    from crescent.core.config import CrescentCfg
    from crescent.core.model import ByteLM
    from crescent.train.utils import encode_bytes

    cfg = CrescentCfg.load(cfg_path)
    sd = torch.load(ckpt_path, map_location="cpu")["model_state"]
    m = ByteLM(
        cfg.model.vocab_size,
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.n_heads,
        cfg.model.d_ff,
        cfg.model.dropout,
        cfg.model.max_seq_len,
    )
    m.load_state_dict(sd, strict=True)
    m.eval()
    text = open(sample_path, "rb").read()[:4096].decode("utf-8", "ignore")
    x = encode_bytes(text)[: cfg.model.max_seq_len + 1]
    inp, tgt = x[:-1].unsqueeze(0), x[1:].unsqueeze(0)
    with torch.no_grad():
        _, loss = m(inp, tgt)
    return float(loss), math.exp(float(loss))


def calc_ppl_bitnet(cfg_path, ckpt_path, sample_path):
    import torch

    from crescent.bitnet.model_bitnet import ByteBitNetLM
    from crescent.core.config import CrescentCfg
    from crescent.train.utils import encode_bytes

    cfg = CrescentCfg.load(cfg_path)
    variant = getattr(cfg.model, "bitnet_variant", "1p58")
    sd = torch.load(ckpt_path, map_location="cpu")["model_state"]
    m = ByteBitNetLM(
        cfg.model.vocab_size,
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.n_heads,
        cfg.model.d_ff,
        cfg.model.dropout,
        cfg.model.max_seq_len,
        variant=variant,
        act_bits=getattr(cfg.model, "act_bits", 8),
        group_size=getattr(cfg.model, "group_size", 64),
        use_rope=getattr(cfg.model, "use_rope", False),
    )
    m.load_state_dict(sd, strict=True)
    m.eval()
    text = open(sample_path, "rb").read()[:4096].decode("utf-8", "ignore")
    x = encode_bytes(text)[: cfg.model.max_seq_len + 1]
    inp, tgt = x[:-1].unsqueeze(0), x[1:].unsqueeze(0)
    with torch.no_grad():
        _, loss = m(inp, tgt)
    return float(loss), math.exp(float(loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-url", required=True)
    ap.add_argument("--bitnet-url", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--dense-slot", default="current")
    ap.add_argument("--bitnet-slot", default="B")
    ap.add_argument("--dense-cfg", default="configs/model/tiny-char-256.yaml")
    ap.add_argument("--bitnet-cfg", default="configs/model/bitnet-b1p58-char-256.yaml")
    ap.add_argument("--data-file", default="data/sample.txt")
    args = ap.parse_args()

    # 端點基準測試
    dense_stats, d_lat, d_tps, _ = bench_endpoint(
        "dense",
        args.dense_url,
        args.prompt,
        args.runs,
        args.warmup,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    bit_stats, b_lat, b_tps, _ = bench_endpoint(
        "bitnet",
        args.bitnet_url,
        args.prompt,
        args.runs,
        args.warmup,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    # 報表目錄
    d_out = os.path.join("slots", args.dense_slot, "runtime", "bench")
    b_out = os.path.join("slots", args.bitnet_slot, "runtime", "bench")
    ensure_dir(d_out)
    ensure_dir(b_out)

    d_csv = dump_report(d_out, "dense", dense_stats, d_lat, d_tps)
    b_csv = dump_report(b_out, "bitnet", bit_stats, b_lat, b_tps)

    # PPL/LOSS（可在服務不跑的環境離線算）
    d_ckpt = os.path.join("slots", args.dense_slot, "weights", "latest.pt")
    # bit ckpt 檔名依 variant；先嘗試 b1.58，失敗再嘗試 1b
    b_ckpt = os.path.join("slots", args.bitnet_slot, "weights", "latest_bitnet_1p58.pt")
    if not os.path.isfile(b_ckpt):
        b_ckpt = os.path.join("slots", args.bitnet_slot, "weights", "latest_bitnet_1b.pt")

    try:
        d_loss, d_ppl = calc_ppl_dense(args.dense_cfg, d_ckpt, args.data_file)
        dense_stats.update({"loss": d_loss, "ppl": d_ppl})
    except Exception as e:
        dense_stats.update({"loss": None, "ppl": None, "loss_error": str(e)})

    try:
        b_loss, b_ppl = calc_ppl_bitnet(args.bitnet_cfg, b_ckpt, args.data_file)
        bit_stats.update({"loss": b_loss, "ppl": b_ppl})
    except Exception as e:
        bit_stats.update({"loss": None, "ppl": None, "loss_error": str(e)})

    # 更新 summary JSON
    with open(os.path.join(d_out, "dense_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dense_stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(b_out, "bitnet_summary.json"), "w", encoding="utf-8") as f:
        json.dump(bit_stats, f, indent=2, ensure_ascii=False)

    # 產圖（若有 pandas/matplotlib）
    common_out = os.path.join("slots", args.bitnet_slot, "runtime", "bench")  # 圖片放這裡
    maybe_plot(common_out, d_csv, b_csv)

    # 簡要列印
    print("\n=== Dense ===")
    print(json.dumps(dense_stats, indent=2, ensure_ascii=False))
    print("\n=== BitNet ===")
    print(json.dumps(bit_stats, indent=2, ensure_ascii=False))
    print(f"\n[OK] CSV/JSON/PNG 已輸出到: {common_out}")


if __name__ == "__main__":
    main()
