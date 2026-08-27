import os

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

PAD_BYTE = 0
SPECIALS = ["<sos>","<eos>", "<pad>"]

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def train_bpe_tokenizer(texts, vocab_size=4000, save_path=None):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
            texts,
            trainer=trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
            )
    if save_path:
        tok.save(save_path)
    return tok

def build_bpe_tokenizer(corpus_path, vocab_size=4000, cache_path=None):
    """Load cached tokenizer, or train on a corpus and cache it."""
    if cache_path and os.path.exists(cache_path):
        return Tokenizer.from_file(cache_path)
    tok = train_bpe_tokenizer(load_lines(corpus_path), vocab_size, save_path=cache_path)
    return tok

def encode_src(tok, line, mode):
    """mode='bpe': learned subword ids over the cipher. mode='bytes': raw byte ids."""
    if mode == "bpe":
        return tok.encode(line).ids
    return list(line.encode("utf-8"))

def encode_tgt(tok, line, mode):
    """mode='bpe': sos + ids + eos. mode='bytes': raw UTF-8 byte ids"""
    if mode == "bpe":
        return [tok.token_to_id("<sos>")] + tok.encode(line).ids + [tok.token_to_id("<eos>")]
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
