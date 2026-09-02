import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, WindowedCausalAttention
from models.transformer import PositionwiseFFN, EncoderBlock, DecoderBlock, Encoder, Decoder

class _LocalEncBlock(nn.Module):
    """windowed-causal self-attn + FFN (BLT local encoder block, paper §3.2)."""

    def __init__(self, cfg, window):
        super().__init__()
        self.attn = WindowedCausalAttention(cfg["d_model"], cfg["n_heads"], window, cfg["dropout"])
        self.norm1 = nn.LayerNorm(cfg["d_model"])
        self.norm2 = nn.LayerNorm(cfg["d_model"])
        self.ffn = PositionwiseFFN(cfg["d_model"], cfg["d_ff"], cfg["dropout"])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class _LocalDecBlock(nn.Module):
    """windowed-causal self-attn + cross-attn to patches + FFN (paper §3.3)."""

    def __init__(self, cfg, window):
        super().__init__()
        self.attn = WindowedCausalAttention(cfg["d_model"], cfg["n_heads"], window, cfg["dropout"])
        self.cross_attn = MultiHeadAttention(cfg["d_model"], cfg["n_heads"],
                                             max_len=cfg["max_len"], dropout=cfg["dropout"])
        self.norm1 = nn.LayerNorm(cfg["d_model"])
        self.norm2 = nn.LayerNorm(cfg["d_model"])
        self.norm3 = nn.LayerNorm(cfg["d_model"])
        self.ffn = PositionwiseFFN(cfg["d_model"], cfg["d_ff"], cfg["dropout"])

    def forward(self, x, enc_out):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), kv=enc_out)
        x = x + self.ffn(self.norm3(x))
        return x

class LocalEncoder(nn.Module):
    """Byte (b, s_bytes) -> patch embeddings (B, s_patches, d_model)."""

    def __init__(self, cfg, patch_size, n_local_layers):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(256, cfg["d_model"])
        self.pos_in_patch = nn.Embedding(patch_size, cfg["d_model"])
        self.blocks = nn.ModuleList([_LocalEncBlock(cfg, cfg["local_window"])
                                     for _ in range(n_local_layers)])

    def forward(self, byte_ids, byte_mask=None):
        b, s = byte_ids.size()
        x = self.byte_embed(byte_ids)
        pos = torch.arange(s, device=byte_ids.device) % self.patch_size
        x = x + self.pos_in_patch(pos)
        # smimpler -- build positions as arange over full byte sequence
        for blk in self.blocks:
            x = blk(x)
        # reshape --> patches, masked mean over each patch's bytes
        n_patch = (s + self.patch_size -1) // self.patch_size
        pad = n_patch * self.patch_size - s
        if pad:
            x = nn.functional.pad(x, (0, 0, 0, pad))
            if byte_mask is not None:
                byte_mask = nn.functional.pad(byte_mask, (0, pad))
        x = x.view(b, n_patch, self.patch_size, -1)
        if byte_mask is None:
            return x.mean(dim=2)
        m = byte_mask.view(b, n_patch, self.patch_size).unsqueeze(-1).float()
        return (x*m).sum(dim=2) / m.sum(dim=2).clamp(min=1)

class LocalDecoder(nn.Module):
    """patch reps (B, s_patches,d) --> byte logits (B, s_bytes, 256)"""

    def __init__(self, cfg, patch_size, n_local_layers):
        super().__init__()
        self.patch_size = patch_size
        self.pos_in_patch = nn.Embedding(patch_size, cfg["d_model"])
        self.blocks = nn.ModuleList([_LocalDecBlock(cfg, cfg["local_window"])
                                     for _ in range(n_local_layers)])
        self.out_proj = nn.Linear(cfg["d_model"], 256)

    def forward(self, patch_reps, enc_out, src_mask=None):
        b, n, d = patch_reps.size()
        x = patch_reps.repeat_interleave(self.patch_size, dim=1)
        pos = torch.arange(self.patch_size, device=x.device).unsqueeze(0).repeat(b, n).view(b, -1)
        x = x + self.pos_in_patch(pos)
        for blk in self.blocks:
            x = blk(x, enc_out)
        return self.out_proj(x)

class BLTModel(nn.Module):
    """bytes-->local enc-->global enc-dec(patch level)-->local dec-->byte logits"""

    def __init__(self, cfg, patch_size=8, n_local_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.local_enc = LocalEncoder(cfg, patch_size, n_local_layers)
        self.local_dec = LocalDecoder(cfg, patch_size, n_local_layers)
        self.global_enc = Encoder(cfg, vocab_size=None)
        self.global_dec = Decoder(cfg, vocab_size=None, with_output_head=False)
        self.bos = nn.Parameter(torch.zeros(cfg["d_model"]))

    def forward(self, src_bytes, tgt_bytes, src_mask=None):
        if src_mask is not None and src_mask.dim() == 4:
            src_mask = src_mask[:, 0, 0]
        src_p = self.local_enc(src_bytes, src_mask)
        enc_out = self.global_enc(src_p)
        tgt_p = self.local_enc(tgt_bytes)
        bos = self.bos.view(1, 1, -1).expand(tgt_p.size(0), 1, -1)
        dec_in = torch.cat([bos, tgt_p[:, :-1]], dim=1)
        reps = self.global_dec(dec_in, enc_out)
        logits = self.local_dec(reps, enc_out)
        return logits[:, : tgt_bytes.size(1)]
