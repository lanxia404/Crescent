import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock


class ByteLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        # x: (B, T) long
        B, T = x.size()
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max {self.max_seq_len}"
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.head(h)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,  # (B, T) byte-ids
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,  # 新增 nucleus sampling
        repetition_penalty: float | None = None,  # 新增 重複抑制係數(>1.0)
        rep_window: int = 256,  # 在最近多少 tokens 計算重複
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """
        以抽樣生成序列（支援 top-k, top-p, 重複抑制）。repetition_penalty >1 會懲罰最近窗口內高頻 token。
        """
        B = idx.size(0)
        device = idx.device
        vocab = getattr(self, "vocab_size", 256)

        for _ in range(max_new_tokens):
            # 只餵模型允許的最大長度（避免超長）
            x = idx[:, -getattr(self, "max_seq_len", idx.shape[1]) :]
            logits, _ = self(x, None) if self.training else self(x, None)  # 適配你 forward 的回傳
            logits = logits[:, -1, :]  # (B, vocab)

            # 重複抑制：對最近 rep_window 內出現過的 token 降溫
            if repetition_penalty and repetition_penalty > 1.0:
                # 統計每個 batch 近期 token 次數
                for b in range(B):
                    recent = idx[b, -rep_window:].tolist()
                    if recent:
                        counts = torch.bincount(
                            torch.tensor(recent, device=device, dtype=torch.long), minlength=vocab
                        ).float()
                        # 依次數扣 logits（log-domain 下線性扣減）
                        logits[b] -= counts * math.log(repetition_penalty)

            # 溫度
            if temperature is not None and temperature > 0:
                logits = logits / max(temperature, 1e-6)

            # top-k
            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

            # top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                # mask 超出機率質量的 token（保留至少一個）
                mask = cumprobs > top_p
                mask[:, 0] = False
                sorted_logits = torch.where(
                    mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits
                )
                # 還原到原索引順序
                inv_idx = torch.argsort(sorted_idx, dim=-1)
                logits = torch.gather(sorted_logits, 1, inv_idx)

            probs = F.softmax(logits, dim=-1)

            # 取樣（若 temperature=0.0，外層會讓 logits 極大，這裡也能改成 argmax）
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)

            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None:
                # 當所有 batch 都抽到 EOS 就提早停止
                if bool((next_id == eos_id).all()):
                    break

        return idx
