---
language: en
library_name: pytorch
pipeline_tag: text2text-generation
license: mit
---
# ANLP Assignment 1 - from-scratch Encoder-Decoder Transformer (C1-C5)

Ablation-study checkpoints produced by `src/train.py`. Implementation (see
`src/`): attention, positional encodings, norms and the BLT modules are
written from scratch, as is the byte-level BPE tokenizer (`bpe_*.json` caches
are the full tokenizer state). Each `.pt` is a bare `state_dict` matching the
constructors in `src/models/`.

| config | bit_accuracy | sequence_accuracy | levenshtein | bleu | params |
|---|---|---|---|---|---|
| C1 | 0.9668 | 0.6560 | 0.6820 | 0.9688 | 6048256 |
| C2 | 0.5520 | 0.0000 | 31.3370 | 0.2849 | 6048256 |
| C3 | 0.9120 | 0.3870 | 2.1490 | 0.8967 | 5163520 |
| C4 | 0.9644 | 0.6300 | 0.7135 | 0.9663 | 6043136 |
| C5 | 0.9996 | 0.9600 | 0.0455 |  | 9407236 |

## Files
| checkpoint | size | date |
|---|---|---|
| C1.pt | 24.8 MB | 2026-09-02 17:49 |
| C2.pt | 24.8 MB | 2026-09-02 18:06 |
| C3.pt | 21.2 MB | 2026-09-02 18:23 |
| C4.pt | 24.7 MB | 2026-09-02 18:40 |
| C5.pt | 38.4 MB | 2026-09-02 18:52 |
