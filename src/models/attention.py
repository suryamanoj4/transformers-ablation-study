import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from models.positional import RotaryEmbedding

class ScaledDotProductAttention(nn.Module):
    """Pure attention calculation: softmax(Q K^T/ sqrt(d_k) V)"""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k  = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        attn = self.dropout(f.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0, "make sure d_model is divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope = RotaryEmbedding(self.d_k, max_len) if use_rope else None
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(kv).view(b, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_proj(kv).view(b, -1, self.n_heads, self.d_k).transpose(1,2)
        if self.rope is not None and kv is None:
            q, k = self.rope.rotate(q, k)
        out = self.attn(q ,k ,v ,mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_len=512, dropout=0.0, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0, "make sure d_model is divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "make sure n_heads is divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.rope = RotaryEmbedding(self.d_k, max_len) if use_rope else None
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(kv).view(b, -1, self.n_kv_heads, self.d_k).transpose(1,2)
        v = self.v_proj(kv).view(b, -1, self.n_kv_heads, self.d_k).transpose(1,2)
        if self.rope is not None and kv is None:
            q, k = self.rope.rotate(q, k)
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        out = self.attn(q ,k ,v ,mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)

class WindowedCausalAttention(nn.Module):
    """Causal sliding-window self-attention for BLT local blocks (paper §3.2):
    position i attends to [max(0, i-w), i] only -> O(S*w) compute and memory."""

    def __init__(self, d_model, n_heads, window, dropout=0.0, chunk=256):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window = window
        self.chunk = chunk
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        w = max(self.window, 1)
        outs = []
        for c0 in range(0, s, self.chunk):
            c1 = min(c0 + self.chunk, s)
            lo = max(0, c0 - w)
            kk = k[:, :, lo:c1]
            vv = v[:, :, lo:c1]
            qq = q[:, :, c0:c1]
            sc = torch.matmul(qq, kk.transpose(-1, -2)) / math.sqrt(self.d_k)
            iq = torch.arange(c0, c1, device=x.device).view(1, 1, -1, 1)
            ik = torch.arange(lo, c1, device=x.device).view(1, 1, 1, -1)
            keep = (ik >= iq - w) & (ik <= iq)
            if mask is not None:
                keep = keep & mask[:, :, c0:c1, lo:c1]
            sc = sc.masked_fill(~keep, float("-inf"))
            attn = self.dropout(sc.softmax(-1))
            outs.append(torch.matmul(attn, vv))
        o = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(o)
