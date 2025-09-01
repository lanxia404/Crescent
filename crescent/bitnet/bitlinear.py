import torch
import torch.nn as nn

from .quant import (
    absmax_quantize_dequant,
    binary_weight_quantize_centered,
    ternary_weight_quantize_absmean,
)


class BitLinear1b(nn.Module):
    """
    1-bit BitLinear（二值權重 + absmax activation）。不使用 bias（與 LLaMA/BitNet 對齊）。
    """

    def __init__(
        self,
        in_features,
        out_features,
        act_bits: int = 8,
        group_size: int | None = 64,
        per_token: bool = True,
        per_row_scale: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.act_bits = act_bits
        self.group_size = group_size
        self.per_token = per_token
        self.per_row_scale = per_row_scale
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        # activation quant (dequant回浮點方便與 PyTorch 其餘層相容)
        x_q, _ = absmax_quantize_dequant(
            x, bits=self.act_bits, group_size=self.group_size, per_token=self.per_token
        )
        Wb = binary_weight_quantize_centered(self.weight, per_row=self.per_row_scale)
        return nn.functional.linear(x_q, Wb, self.bias)


class BitLinear158(nn.Module):
    """
    1.58-bit（三值）BitLinear：權重 {-1,0,1}，activation 延續 absmax；無 bias。
    """

    def __init__(
        self,
        in_features,
        out_features,
        act_bits: int = 8,
        group_size: int | None = 64,
        per_token: bool = True,
        per_row_absmean: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.act_bits = act_bits
        self.group_size = group_size
        self.per_token = per_token
        self.per_row_absmean = per_row_absmean
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x_q, _ = absmax_quantize_dequant(
            x, bits=self.act_bits, group_size=self.group_size, per_token=self.per_token
        )
        Wt = ternary_weight_quantize_absmean(self.weight, per_row=self.per_row_absmean)
        return nn.functional.linear(x_q, Wt, self.bias)
