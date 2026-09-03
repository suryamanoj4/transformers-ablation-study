import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization (from scratch), statistics in fp32 even under
    fp16 autocast - mixed-precision norm statistics otherwise silently lose
    accuracy."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        var = x32.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x32 - mean) * torch.rsqrt(var + self.eps)
        out = self.gamma.float() * x_hat + self.beta.float()
        return out.to(in_dtype)


class RMSNorm(nn.Module):
    """RMS normalization (from scratch), fp32 statistics as in LayerNorm."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.float()
        rms_inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = self.gamma.float() * (x32 * rms_inv)
        return out.to(in_dtype)