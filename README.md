# ANLP Assignment 1 — Encoder–Decoder Transformer with Ablation (C1–C5)

From-scratch encoder–decoder Transformer + a five-configuration ablation
(C1–C5) that decodes an XOR-cipher binary stream into plaintext. All model
modules (attention, positional encodings, norms, BLT) and the byte-level BPE
tokenizer are implemented from scratch — no `nn.Transformer`, no tokenizer
library.

## Structure

```
src/
  models/
    attention.py    # MHA and GQA
    positional.py   # Sinusoidal and RoPE
    norm.py         # LayerNorm and RMSNorm
    blt.py          # Local Encoder/Decoder for BLT (C5)
  dataset.py        # Windowed (tokenized & token-free) loaders, from-scratch BPE
  train.py          # Main training loop with WandB, C1-C5 config registry
  utils.py          # Metrics and plots
output/             # Saved plots, checkpoints, results
report/             # LaTeX report (report.tex -> report.pdf)
tools/
  upload_hf.py      # Upload checkpoints to Hugging Face Hub
```

## Reproducing the results

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Data

Place the dataset `Dataset_A1/` (containing `brown_cipher.txt` /
`brown_plain.txt`, 5,000 aligned pairs) in the repository root.
(`Dataset_A1/` is `.gitignore`d.)

### 3. Train and run

The tokenizers train once and are cached in `outputs/`; all configurations
share the same hyperparameters (d=256, 3+3 layers, 32-char phase-aligned
windows, batch 64, 32 epochs, lr 8e-4 + warmup/cosine, wd 0.01, grad clip
1.0, fp16 AMP, best-val checkpoint evaluated on the test set).

```bash
# full ablation (all five configs), with WandB
python src/train.py --configs all

# run specific configs
python src/train.py --configs C1,C2

# quick sanity run (few steps, no WandB)
python src/train.py --configs C1 --smoke

# skip WandB
python src/train.py --configs all --no-wandb
```

WandB auth: `WANDB_API_KEY=...` in `.env` (gitignored, auto-loaded), or
`wandb login`.

### 4. Outputs

Everything lands in `outputs/`: `ablation_results.json` (test metrics +
compute profile), `histories.json`, `loss_curves.png`, `metrics_bar.png`,
`C1.pt…C5.pt` (best checkpoints), and the BPE caches
(`bpe_1024.json`, `bpe_cipher_1024.json`).

### 5. Report

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

Produces `report.pdf` (6 body pages + references).

### 6. Uploading to Hugging Face

```bash
export HF_TOKEN=hf_xxx          # or put HF_TOKEN=... in .env
python tools/upload_hf.py --repo-id <you>/anlp-assignment-1
python tools/upload_hf.py --private      # private repository
```

## Notes

- The cipher is a period-8 XOR over bytes; 32-char windows are multiples of
  the key period, so every window is phase-aligned.
- Greedy decoding is used for all reported metrics.
- Checkpoints are bare `state_dict`s: load with
  `build_model(cfg, tgt_vocab, src_vocab)` from `src/train.py`, then
  `model.load_state_dict(torch.load("outputs/C1.pt"))`.
- nltk's edit-distance guard is raised in `src/utils.py` for long strings.
