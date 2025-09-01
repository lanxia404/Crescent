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
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k: int | None = None):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            T = idx.size(1)
            idx_cond = idx[:, -self.max_seq_len :] if T > self.max_seq_len else idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
