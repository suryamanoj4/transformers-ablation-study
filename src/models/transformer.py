import math

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.positional import SinusoidalEmbeddings
from models.norm import LayerNorm, RMSNorm

def _build_attn(cfg):
    """one attention block , chosen by cfg for MHA or GQA"""
    if cfg["attention"] == "gqa":
        return GroupedQueryAttention(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"],
                                     max_len=cfg["max_len"], dropout=cfg["dropout"],
                                     use_rope=cfg["use_rope"])

    return MultiHeadAttention(cfg["d_model"], cfg["n_heads"],
                              max_len=cfg["max_len"], dropout=cfg["dropout"],
                              use_rope=cfg["use_rope"])

def _build_norm(cfg):
    """one nrmalization block by cfg LayerNorm and RMSNorm"""
    return RMSNorm if cfg["norm"] == "rmsnorm" else LayerNorm

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(f.gelu(self.fc1(x))))

class EncoderBlock(nn.Module):
    """pre norm block"""

    def __init__(self, cfg):
        super().__init__()
        self.attn = _build_attn(cfg)
        norm = _build_norm(cfg)
        self.norm1 = norm(cfg["d_model"])
        self.norm2 = norm(cfg["d_model"])
        self.ffn = PositionwiseFFN(cfg["d_model"], cfg["d_ff"], cfg["dropout"])

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """Pre norm block: casual self attn , cross attn with encode ,fnn"""

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = _build_attn(cfg)
        self.cross_attn = _build_attn(cfg)
        norm = _build_norm(cfg)
        self.norm1 = norm(cfg["d_model"])
        self.norm2 = norm(cfg["d_model"])
        self.norm3 = norm(cfg["d_model"])
        self.ffn = PositionwiseFFN(cfg["d_model"], cfg["d_ff"], cfg["dropout"])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        s = x.size(1)
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=x.device))
        causal = causal.unsqueeze(0).unsqueeze(0)
        self_mask = causal if tgt_mask is None else causal & tgt_mask
        x = x + self.self_attn(self.norm1(x), mask=self_mask)
        x = x + self.cross_attn(self.norm2(x), kv=enc_out, mask=src_mask)
        x = x + self.ffn(self.norm3(x))
        return x

class Encoder(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg["d_model"]) if vocab_size is not None else nn.Identity()
        if vocab_size is not None:
            nn.init.xavier_uniform_(self.embed.weight)
        self.emb_scale = math.sqrt(cfg["d_model"]) if vocab_size is not None else 1.0
        self.pos = (SinusoidalEmbeddings(cfg["d_model"], cfg["max_len"])
                    if not cfg["use_rope"] else nn.Identity())
        self.dropout = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = _build_norm(cfg)(cfg["d_model"])

    def forward(self, src, src_mask=None):
        x = self.dropout(self.pos(self.embed(src) * self.emb_scale))
        for blk in self.blocks:
            x = blk(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, cfg, vocab_size=None, with_output_head=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg["d_model"]) if vocab_size is not None else nn.Identity()
        if vocab_size is not None:
            nn.init.xavier_uniform_(self.embed.weight)
        self.emb_scale = math.sqrt(cfg["d_model"]) if vocab_size is not None else 1.0
        self.pos = (SinusoidalEmbeddings(cfg["d_model"], cfg["max_len"])
                    if not cfg["use_rope"] else nn.Identity())
        self.dropout = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = _build_norm(cfg)(cfg["d_model"])
        self.out_proj = nn.Linear(cfg["d_model"], vocab_size) if with_output_head else nn.Identity()
        if with_output_head and vocab_size is not None:
            self.out_proj.weight = self.embed.weight  # weight tying (Vaswani et al. §3.4)
            nn.init.xavier_uniform_(self.embed.weight)  # tied head: use embed-scale init

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos(self.embed(tgt) * self.emb_scale))
        for blk in self.blocks:
            x = blk(x, enc_out, src_mask, tgt_mask)
        return self.out_proj(self.norm(x))

class EncoderDecoder(nn.Module):
    """Full Model. cfg keys: d_model, d_ff, n_heads, n_kv_heads, n_layers, dropout,
    max_len, use_rope, attention ('mha'|'gqa'), norm ('layernorm'|'rmsnorm')"""

    def __init__(self, cfg, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.encoder = Encoder(cfg, src_vocab_size)
        self.decoder = Decoder(cfg, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)
