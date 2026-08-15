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
