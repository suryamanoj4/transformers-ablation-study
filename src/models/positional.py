import math

import torch
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    """additive positional table: encoder/decoder do 'x + self.pe[:, :seq_len]'."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(d_model // 2)
        theta = pos / 10000 ** (2 * i / d_model)
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

def _rotate_half(x):
    """swap  + negate the second half: [x2; -x1] --> used to rotate pairs"""
    d=x.size(-1)
    x1, x2 =x[..., : d //2], x[..., d//2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """Precompute cos/sin once; buffer follows models device"""

    def __init__(self, d_k, max_len=512, base=10000.0):
        super().__init__()
        assert d_k % 2 == 0
        i = torch.arange(d_k // 2)
        freqs = 1.0 / (base ** (2 * i / d_k))
        t = torch.arange(max_len)
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("cos", torch.cos(angles).repeat_interleave(2, dim=-1).view(1, 1, max_len, d_k))
        self.register_buffer("sin", torch.sin(angles).repeat_interleave(2, dim=-1).view(1, 1, max_len, d_k))

    def rotate(self, q, k):
        s = q.size(2)
        cos = self.cos[:, :, :s].to(q.dtype)
        sin = self.sin[:, :, :s].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
