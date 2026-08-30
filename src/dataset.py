import os

import torch
from torch.utils.data import Dataset
import heapq, json
import numpy as np

PAD_BYTE = 0
SPECIALS = ["<sos>","<eos>", "<pad>"]
N_SPECIALS = len(SPECIALS)
BYTE_OFFSET = N_SPECIALS #specils o,...,2; byte b --> id BYTE_OFFSET + b
N_BASE = BYTE_OFFSET + 256 # first merged token id
KEY = 4096 # key(a,b) = a*KEY + b (vocab < 4096)

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

class ByteLevelBPE:
    def __init__(self, specials, merges):
        self.specials = list(specials)
        self.special_ids = {s: i for i,s in enumerate(specials)}
        self.merges = [tuple(m) for m in merges]
        self.pieces = ([b""] * N_SPECIALS + [bytes([b]) for b in range(256)]
                       + [b""] * len(self.merges))
        for i, (l, r) in enumerate(self.merges):
            self.pieces[N_BASE + i] = self.pieces[l] + self.pieces[r]
        self.rank = {p: r for r, p in enumerate(self.merges)}
        self.vocab_size = N_BASE + len(self.merges)


    def get_vocab_size(self):
        return self.vocab_size

    def token_to_id(self, tok):
        return self.special_ids[tok]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")

    def encode(self, text):
        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        n = len(ids)
        if n < 2:
            return ids
        INF = 1 << 60
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(0, n - 1))
        alive = [True] * n
        head = 0
        heap = [(self.rank.get((ids[i], ids[i + 1]), INF), i, ids[i], ids[i + 1])
                for i in range(n - 1)]
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            if r >= INF or not alive[i]:
                continue
            j = nxt[i]
            if j < 0 or ids[i] != a or ids[j] != b:
                continue
            k = len(ids)
            ids.append(N_BASE + r)
            alive.append(True)
            p = prv[i]
            q = nxt[j]
            prv.append(p)
            nxt.append(q)
            if p >= 0:
                nxt[p] = k
            else:
                head = k
            if q >= 0:
                prv[q] = k
            alive[i] = alive[j] = False
            if p >= 0:
                heapq.heappush(heap, (self.rank.get((ids[p], ids[k]), INF),
                                      p, ids[p], ids[k]))
            if q >= 0:
                heapq.heappush(heap, (self.rank.get((ids[k], ids[q]), INF),
                                      k, ids[k], ids[q]))
        out = []
        i = head
        while i >= 0:
            out.append(ids[i])
            i = nxt[i]
        return out

def train_bpe_tokenizer(texts, vocab_size):
    arrs = [np.array([BYTE_OFFSET + b for b in t.encode("utf-8")], dtype=np.int64)
            for t in texts if t]
    f = np.concatenate([np.append(a,-1) for a in arrs]) if arrs else np.array([], dtype=np.int64)
    merges = []
    while N_BASE + len(merges) < vocab_size and f.size > 2:
        left, right = f[:-1], f[1:]
        valid = (left >= 0) & (right >= 0)
        cnt = np.bincount(np.where(valid, left*KEY + right, 0))
        if cnt.size <= 1 or cnt[1:].max() == 0:
            break
        best = 1 +int(np.argmax(cnt[1:]))
        a, b = best // KEY, best % KEY
        occ = np.nonzero((left==a) & (right == b))[0]
        if occ.size == 0:
            break
        if a == b:
            kept = [occ[0]]
            for p in occ[1:]:
                if p > kept[-1] +1:
                    kept.append(p)
            occ = np.array(kept, dtype=np.int64)
        merges.append((int(a), int(b)))
        mask = np.ones(f.size, dtype=bool)
        mask[occ + 1] = False
        nf = f[mask]
        nf[occ - np.arange(occ.size)] = N_BASE + len(merges) - 1
        f = nf
    return ByteLevelBPE(SPECIALS, merges)

def build_bpe_tokenizer(corpus_path, vocab_size=4000, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                data = json.load(fh)
            if data.get("specials") == SPECIALS and isinstance(data.get("merges"), list):
                return ByteLevelBPE(data["specials"], data["merges"])
        except (ValueError, KeyError):
            pass
    tok = train_bpe_tokenizer(load_lines(corpus_path), vocab_size)
    if cache_path:
        with open(cache_path, "w") as fh:
            json.dump({"specials": SPECIALS, "merges": tok.merges}, fh)
    return tok

def encode_src(tok, line, mode):
    """mode='bpe': learned subword ids over the cipher. mode='bytes': raw byte ids."""
    if mode == "bpe":
        return tok.encode(line)
    return list(line.encode("utf-8"))

def encode_tgt(tok, line, mode):
    """mode='bpe': sos + ids + eos. mode='bytes': raw UTF-8 byte ids"""
    if mode == "bpe":
        return [tok.token_to_id("<sos>")] + tok.encode(line) + [tok.token_to_id("<eos>")]
    return list(line.encode("utf-8"))

class CipherDataset(Dataset):
    def __init__(self, cipher_path, plain_path, src_tokenizer, tgt_tokenizer,
                 mode="bpe", max_len=512):
        self.ciphers = load_lines(cipher_path)
        self.plains = load_lines(plain_path)
        assert len(self.ciphers) == len(self.plains)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.mode = mode
        self.max_len = max_len

    def __len__(self):
        return len(self.ciphers)

    def __getitem__(self, i):
        src = encode_src(self.src_tokenizer, self.ciphers[i], self.mode)[: self.max_len]
        tgt = encode_tgt(self.tgt_tokenizer, self.plains[i], self.mode)[: self.max_len]
        return {"src": torch.tensor(src, dtype=torch.long),
                "tgt": torch.tensor(tgt, dtype=torch.long)}

def collate(batch, pad_tgt_id=0):
    """pad batch; masks are (B,1,1,S) bol, True = real token"""
    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    B, S, T = len(batch), max(s.size(0) for s in srcs), max(t.size(0) for t in tgts)

    src = torch.zeros(B, S, dtype=torch.long)
    tgt = torch.full((B, T), pad_tgt_id, dtype=torch.long)
    src_mask = torch.zeros(B,1,1,S,dtype=torch.bool)
    tgt_mask = torch.zeros(B,1,1,T, dtype=torch.bool)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : s.size(0)] = s
        tgt[i, : t.size(0)] = t
        src_mask[i,0,0,:s.size(0)] = True
        tgt_mask[i,0,0,:t.size(0)] = True

    return {"src":src, "tgt":tgt, "src_mask":src_mask, "tgt_mask":tgt_mask}

def bpe_decode(tok, ids):
    """strips specials, return plain text"""
    specials = {tok.token_to_id(s) for s in SPECIALS}
    return tok.decode([i for i in ids if i not in specials])

def bytes_to_text(ids):
    """byte ids --> text"""
    return bytes(ids).decode("utf-8", errors="replace")
