import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import RMSNorm  # 沿用既有實作
from .bitlinear import BitLinear1b, BitLinear158
from .rope import apply_rope


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 典型做法：兩條線路 -> gate * up，最後投影回 d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SwiGLU_Bit(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        variant: str = "1b",
        act_bits: int = 8,
        gsize: int | None = 64,
    ):
        super().__init__()
        Linear = BitLinear1b if variant == "1b" else BitLinear158
        self.w1 = Linear(d_model, d_ff, act_bits=act_bits, group_size=gsize, bias=False)
        self.w2 = Linear(d_model, d_ff, act_bits=act_bits, group_size=gsize, bias=False)
        self.w3 = Linear(d_ff, d_model, act_bits=act_bits, group_size=gsize, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MHSA_Bit(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        variant: str = "1b",
        act_bits: int = 8,
        gsize: int | None = 64,
        use_rope: bool = False,
        rotary_dim: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        Linear = BitLinear1b if variant == "1b" else BitLinear158
        self.q_proj = Linear(d_model, d_model, act_bits=act_bits, group_size=gsize, bias=False)
        self.k_proj = Linear(d_model, d_model, act_bits=act_bits, group_size=gsize, bias=False)
        self.v_proj = Linear(d_model, d_model, act_bits=act_bits, group_size=gsize, bias=False)
        self.o_proj = Linear(d_model, d_model, act_bits=act_bits, group_size=gsize, bias=False)
        self.drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.d_head
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(), persistent=False
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(0, T, device=x.device)
            q, k = apply_rope(q, k, pos, rotary_dim=self.rotary_dim)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )  # PyTorch 2.x SDPA
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return self.drop(y)


class TransformerBlock_Bit(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        variant: str = "1b",
        act_bits: int = 8,
        gsize: int | None = 64,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHSA_Bit(
            d_model,
            n_heads,
            dropout,
            max_seq_len,
            variant=variant,
            act_bits=act_bits,
            gsize=gsize,
            use_rope=use_rope,
        )
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU_Bit(d_model, d_ff, variant=variant, act_bits=act_bits, gsize=gsize)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
