import torch


def _rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, seq_positions, rotary_dim: int):
    """
    q,k: (B,H,T,C); 只對前 rotary_dim 維套用 RoPE
    seq_positions: (T,)
    """
    if rotary_dim % 2 != 0:
        rotary_dim = rotary_dim - 1
    freqs = torch.arange(0, rotary_dim, 2, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (10000 ** (freqs / rotary_dim))
    t = seq_positions.to(q.dtype)[:, None] * inv_freq[None, :]
    cos = torch.cos(t).repeat_interleave(2, dim=1)  # (T,rotary_dim)
    sin = torch.sin(t).repeat_interleave(2, dim=1)
    # 擴展到 (B,H,T,rotary_dim)
    cos = cos[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
    sin = sin[None, None, :, :].expand_as(cos)

    def rope(x):
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim:]
        x1 = x1 * cos + _rotate_half(x1) * sin
        return torch.cat([x1, x2], dim=-1)

    return rope(q), rope(k)
