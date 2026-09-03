import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention, WindowedCausalAttention
from models.positional import SinusoidalEmbeddings
from models.transformer import PositionwiseFFN, Encoder, Decoder, init_model_params

# byte values are stored with this offset so ids never collide with pad=0;
# ids: 0=pad, 1=bos, 2=eos, 3=unused, 4..259=bytes, vocab=260
BYTE_OFFSET = 4
BOS_ID = 1
EOS_ID = 2
BYTE_VOCAB = 256 + BYTE_OFFSET


def _starts_to_segments(starts, valid):
    """(B,L) patch-start flags -> (seg_per_byte, byte_of, patch_ok, n_patches)."""
    b, l = starts.shape
    st = starts & valid
    st[:, 0] = valid[:, 0] | st[:, 0]
    seg = st.long().cumsum(1) - 1
    seg = seg.clamp(min=0)
    j = max(int(st.sum(1).max().item()), 1)
    seg = seg.clamp(max=j - 1)
    ar = torch.arange(j, device=starts.device).view(1, j, 1)
    byte_of = (seg.unsqueeze(1) == ar) & valid.unsqueeze(1)
    return seg, byte_of.unsqueeze(1), byte_of.any(-1)[:, None, None, :], j


class PatchPool(nn.Module):
    """Cross-attention pooling of a patch's byte states; mean pooling as residual."""

    def __init__(self, d_local, d_global, n_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_local) * 0.02)
        self.attn = MultiHeadAttention(d_local, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_local)
        self.proj = nn.Linear(d_local, d_global)

    def forward(self, byte_states, byte_of, n_patches):
        b = byte_states.size(0)
        q = self.query.expand(b, n_patches, -1)
        pooled = self.attn(q, self.norm(byte_states), mask=byte_of)
        w = byte_of.squeeze(1).float()
        mean = torch.bmm(w, byte_states) / w.sum(-1, keepdim=True).clamp(min=1.0)
        return self.proj(pooled + mean)


class _LocalBlock(nn.Module):
    """windowed self-attn (+ optional cross-attn to memory) + FFN."""

    def __init__(self, d_local, n_heads, d_ff, dropout, window, causal, with_cross=False):
        super().__init__()
        self.attn = WindowedCausalAttention(d_local, n_heads, window, dropout, causal=causal)
        self.cross_attn = MultiHeadAttention(d_local, n_heads, dropout=dropout) \
            if with_cross else None
        self.norm1 = nn.LayerNorm(d_local)
        self.norm2 = nn.LayerNorm(d_local)
        self.norm3 = nn.LayerNorm(d_local)
        self.ffn = PositionwiseFFN(d_local, d_ff, dropout)

    def forward(self, x, mem=None, cross_mask=None):
        x = x + self.attn(self.norm1(x))
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm2(x), kv=mem, mask=cross_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class LocalByteEncoder(nn.Module):
    """byte ids -> windowed byte states -> cross-attention pooled patch reps."""

    def __init__(self, cfg, n_local_layers, local_heads, window, causal):
        super().__init__()
        d_local = cfg["d_model"]
        self.byte_emb = nn.Embedding(BYTE_VOCAB, d_local)
        self.pos = SinusoidalEmbeddings(d_local, cfg["max_len_bytes"])
        self.blocks = nn.ModuleList([
            _LocalBlock(d_local, local_heads, cfg["d_ff"], cfg["dropout"], window, causal)
            for _ in range(n_local_layers)])
        self.norm_out = nn.LayerNorm(d_local)
        self.pool = PatchPool(d_local, cfg["d_model"], local_heads, cfg["dropout"])

    def forward(self, byte_ids, starts):
        valid = byte_ids != 0
        x = self.pos(self.byte_emb(byte_ids))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        seg, byte_of, patch_ok, j = _starts_to_segments(starts, valid)
        patches = self.pool(x, byte_of, j)
        return patches, patch_ok, x, seg


class LocalByteDecoder(nn.Module):
    """Byte decoder with three conditionings: own patch state (gathered),
    patch history (cross-attn) and source byte states (cross-attn)."""

    def __init__(self, cfg, n_local_layers, local_heads, window):
        super().__init__()
        d_local = cfg["d_model"]
        self.byte_emb = nn.Embedding(BYTE_VOCAB, d_local)
        self.pos = SinusoidalEmbeddings(d_local, cfg["max_len_bytes"])
        self.patch_to_byte = nn.Linear(cfg["d_model"], d_local)
        self.fuse = nn.Linear(2 * d_local, d_local)
        self.patch_to_mem = nn.Linear(cfg["d_model"], d_local)
        self.src_to_mem = nn.Linear(d_local, d_local)
        self.blocks = nn.ModuleList([
            _LocalBlock(d_local, local_heads, cfg["d_ff"], cfg["dropout"], window,
                        causal=True, with_cross=True) for _ in range(n_local_layers)])
        self.norm_out = nn.LayerNorm(d_local)
        self.out_proj = nn.Linear(d_local, BYTE_VOCAB)

    def forward(self, byte_ids, patch_states, patch_ok, seg_ids,
                src_byte_states=None, src_byte_mask=None):
        b, l = byte_ids.shape
        j = patch_states.size(1)
        valid = (byte_ids != 0)[:, None, None, :]

        self_mask = valid.expand(b, 1, l, l)
        ar = torch.arange(j, device=byte_ids.device).view(1, 1, 1, j)
        patch_cross = (ar <= seg_ids.view(b, 1, l, 1)) & patch_ok
        mems = [self.patch_to_mem(patch_states)]
        cross = patch_cross
        if src_byte_states is not None:
            s = src_byte_states.size(1)
            ok = src_byte_mask if src_byte_mask is not None else \
                torch.ones(b, 1, 1, s, dtype=torch.bool, device=byte_ids.device)
            mems.append(self.src_to_mem(src_byte_states))
            cross = torch.cat([cross, ok.expand(b, 1, l, s)], dim=-1)
        mem = torch.cat(mems, dim=1)

        own = self.patch_to_byte(patch_states.gather(
            1, seg_ids.unsqueeze(-1).expand(-1, -1, patch_states.size(-1))))
        e = self.pos(self.byte_emb(byte_ids))
        x = self.fuse(torch.cat([e, own], dim=-1))

        for blk in self.blocks:
            x = blk(x, mem, cross)
        return self.out_proj(self.norm_out(x))


class BLTModel(nn.Module):
    """Token-free encoder-decoder with fixed-width byte patching (C5)."""

    def __init__(self, cfg, patch_size, n_local_layers):
        super().__init__()
        self.patch_size = patch_size
        self.local_enc_src = LocalByteEncoder(cfg, n_local_layers, cfg["local_heads"],
                                              cfg["local_window"], causal=False)
        self.local_enc_tgt = LocalByteEncoder(cfg, n_local_layers, cfg["local_heads"],
                                              cfg["local_window"], causal=True)
        self.local_dec = LocalByteDecoder(cfg, n_local_layers, cfg["local_heads"],
                                          cfg["local_window"])
        self.global_enc = Encoder(cfg, vocab_size=None)
        self.global_dec = Decoder(cfg, vocab_size=None, with_output_head=False)
        self.start_patch = nn.Parameter(torch.zeros(1, 1, cfg["d_model"]))
        nn.init.normal_(self.start_patch, std=0.02)
        init_model_params(self, pad_id=0)

    def _starts(self, byte_ids):
        b, l = byte_ids.shape
        starts = torch.zeros(b, l, dtype=torch.bool, device=byte_ids.device)
        starts[:, :: self.patch_size] = True
        return starts

    def encode(self, src_bytes):
        starts = self._starts(src_bytes)
        patches, patch_ok, src_states, _ = self.local_enc_src(src_bytes, starts)
        memory = self.global_enc(patches, src_mask=patch_ok)
        src_byte_mask = (src_bytes != 0)[:, None, None, :]
        return memory, patch_ok, src_states, src_byte_mask

    def decode(self, tgt_in_bytes, memory, patch_ok, src_states=None, src_byte_mask=None):
        starts = self._starts(tgt_in_bytes)
        tgt_patches, tgt_ok, _, seg = self.local_enc_tgt(tgt_in_bytes, starts)
        b, j = tgt_patches.size(0), tgt_patches.size(1)
        start = self.start_patch.expand(b, 1, -1)
        pad = tgt_in_bytes.new_zeros(b, 1)
        p_in = torch.cat([start, tgt_patches[:, :-1]], dim=1)
        ok_in = torch.cat([torch.ones_like(tgt_ok[..., :1]), tgt_ok[..., :-1]], dim=-1)
        causal = torch.tril(torch.ones(j, j, dtype=torch.bool, device=tgt_in_bytes.device))
        self_mask = ok_in & causal[None, None]
        g = self.global_dec(p_in, memory, src_mask=None, tgt_mask=self_mask)
        return self.local_dec(tgt_in_bytes, g, ok_in, seg, src_states, src_byte_mask)

    def forward(self, src_bytes, tgt_bytes, src_mask=None):
        memory, patch_ok, src_states, src_byte_mask = self.encode(src_bytes)
        logits = self.decode(tgt_bytes, memory, patch_ok, src_states, src_byte_mask)
        return logits[:, : tgt_bytes.size(1)]
