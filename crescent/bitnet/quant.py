import torch


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g  # straight-through


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, g):
        return g.clamp_(-1, 1)


def absmax_quantize_dequant(
    x: torch.Tensor,
    bits: int,
    group_size: int | None = None,
    per_token: bool = True,
    eps: float = 1e-8,
):
    """
    將 activation 以 absmax 量化後立刻反量化回浮點（便於在 PyTorch 做 QAT）。
    回傳 dequant_x（浮點）與 scale（可選）。
    - per_token=True: 於 (B,T,*) 每個 token 計算 absmax
    - group_size: 針對最後一維做 group quantization（如 64）
    公式參考 BitNet：xe = Clip(x * Qb / gamma, -Qb+eps, Qb-eps)，再除回 (Qb/gamma)
    """
    assert bits >= 2 and bits <= 16
    Qb = 2 ** (bits - 1) - 1  # [-Qb, +Qb]
    shape = x.shape
    last = shape[-1]
    x_view = x.view(-1, last)  # 合併 B,T 維度，便於 per-token

    if group_size is None:
        groups = 1
        reshaped = x_view.unsqueeze(1)  # (N,1,C)
    else:
        assert last % group_size == 0
        groups = last // group_size
        reshaped = x_view.view(-1, groups, group_size)  # (N,G,Gs)

    if per_token:
        gamma = reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(eps)  # (N,G,1)
    else:
        gamma = (
            reshaped.abs().amax(dim=(0, -1), keepdim=True).expand_as(reshaped).clamp_min(eps)
        )  # 全域/每組

    y = reshaped * (Qb / gamma)
    y = torch.clamp(y, -Qb + 1e-6, Qb - 1e-6)
    q = RoundSTE.apply(y)
    deq = q * (gamma / Qb)

    deq = deq.view(*x_view.shape) if group_size else deq.squeeze(1)
    return deq.view(*shape), gamma


def binary_weight_quantize_centered(W: torch.Tensor, per_row: bool = True, eps: float = 1e-8):
    """
    BitNet(2023) 的二值權重：先中心化再 sign，並乘 beta（absmean）作縮放。
    W_bin = sign(W - mean(W)) * beta
    beta 預設為中心化後 |W| 的均值（可 per-row）
    """
    if per_row:
        mean = W.mean(dim=1, keepdim=True)
        centered = W - mean
        beta = centered.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    else:
        mean = W.mean()
        centered = W - mean
        beta = centered.abs().mean().clamp_min(eps)
    Wb = SignSTE.apply(centered) * beta
    return Wb


def ternary_weight_quantize_absmean(W: torch.Tensor, per_row: bool = True, eps: float = 1e-8):
    """
    BitNet b1.58 的三值權重：先除以 absmean，再四捨五入到 {-1,0,1}（absmean quantization）。
    Wt = round( W / mean(|W|) ) 夾在 [-1,1]
    注意：此處直接回傳三值權重（不再乘回 scale），符合論文描述的「absmean 後取整」精神。
    """
    if per_row:
        scale = W.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    else:
        scale = W.abs().mean().clamp_min(eps)
    normed = W / scale
    Wt = torch.clamp(RoundSTE.apply(normed), -1.0, 1.0)
    return Wt
