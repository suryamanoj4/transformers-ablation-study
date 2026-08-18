import argparse
import os
import resource
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import (CipherDataset, bpe_decode, build_bpe_tokenizer,
                     bytes_to_text, collate)
from models.blt import BLTModel
from models.transformer import EncoderDecoder
import utils

OUT_DIR = "outputs"


def load_dotenv(path=".env"):
    """Minimal .env loader: sets KEY=VALUE pairs into os.environ (no overwrite)."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

BASE = dict(d_model=128, d_ff=512, n_heads=4, n_kv_heads=1, n_layers=4,
            dropout=0.1, max_len=512, eval_max_len=512, patch_size=8,
            n_local_layers=2, src_vocab=256, tgt_vocab=4000,
            batch_size=16, lr=1e-3, epochs=5, log_every=10,
            train_frac=0.8, val_frac=0.1, seed=42,
            data_dir="Dataset_A1", cipher_file="brown_cipher.txt",
            plain_file="brown_plain.txt")

CONFIGS = {
    "C1": dict(BASE, attention="mha",    norm="layernorm", use_rope=False, tokenization="bpe"),
    "C2": dict(BASE, attention="mha",    norm="layernorm", use_rope=True,  tokenization="bpe"),
    "C3": dict(BASE, attention="gqa",    norm="layernorm", use_rope=False, tokenization="bpe"),
    "C4": dict(BASE, attention="mha",    norm="rmsnorm",   use_rope=False, tokenization="bpe"),
    "C5": dict(BASE, attention="mha",    norm="layernorm", use_rope=False, tokenization="bytes"),
}


def build_model(cfg, tgt_vocab):
    model_cfg = dict(cfg, attention=cfg["attention"], norm=cfg["norm"],
                     use_rope=cfg["use_rope"])
    if cfg["tokenization"] == "bytes":
        return BLTModel(model_cfg, cfg["patch_size"], cfg["n_local_layers"])
    return EncoderDecoder(model_cfg, src_vocab_size=cfg["src_vocab"],
                          tgt_vocab_size=tgt_vocab)


def make_loaders(cfg, tokenizer):
    mode = cfg["tokenization"]
    pad = 0 if mode == "bytes" else tokenizer.token_to_id("<pad>")
    ds = CipherDataset(os.path.join(cfg["data_dir"], cfg["cipher_file"]),
                       os.path.join(cfg["data_dir"], cfg["plain_file"]),
                       tokenizer, mode=mode, max_len=cfg["max_len"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    n = len(ds)
    n_tr = int(n * cfg["train_frac"])
    n_va = int(n * cfg["val_frac"])
    tr, va, te = random_split(ds, [n_tr, n_va, n - n_tr - n_va], gen)
    mk = lambda d: DataLoader(d, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=lambda b: collate(b, pad_tgt_id=pad))
    return mk(tr), mk(va), mk(te)


def train_step(model, opt, batch, cfg):
    src, tgt, sm, tm = batch["src"], batch["tgt"], batch["src_mask"], batch["tgt_mask"]
    if cfg["tokenization"] == "bytes":
        logits = model(src, tgt, sm)
        labels = tgt.clone().masked_fill(~tm[:, 0, 0], -100)
    else:
        logits = model(src, tgt[:, :-1], sm, tm[:, :, :, :-1])
        labels = tgt[:, 1:].masked_fill(~tm[:, 0, 0, 1:], -100)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def greedy_decode(model, src, sm, cfg, tokenizer, max_len):
    """C1-C4: token loop. C5: patch loop (patch_size bytes per step)."""
    device = next(model.parameters()).device
    src = src.to(device)
    sm = sm.to(device)
    model.eval()
    if cfg["tokenization"] == "bytes":
        enc_out = model.global_enc(model.local_enc(src, sm[:, 0, 0]))
        bos = model.bos.view(1, 1, -1).expand(src.size(0), 1, -1)
        cur, out = bos, []
        for _ in range((max_len + cfg["patch_size"] - 1) // cfg["patch_size"]):
            reps = model.global_dec(cur, enc_out)
            patch = model.local_dec(reps[:, -1:], enc_out).argmax(-1)
            out.append(patch)
            cur = torch.cat([cur, model.local_enc(patch)], dim=1)
        return torch.cat(out, dim=1)[:, :max_len]
    enc_out = model.encoder(src, sm)
    bos_id, eos_id = tokenizer.token_to_id("<sos>"), tokenizer.token_to_id("<eos>")
    tgt = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        nxt = model.decoder(tgt, enc_out, src_mask=sm)[:, -1].argmax(-1).unsqueeze(-1)
        tgt = torch.cat([tgt, nxt], dim=1)
        if (nxt == eos_id).all():
            break
    return tgt[:, 1:]


def evaluate(model, loader, cfg, tokenizer, max_len=None, max_batches=None):
    model.eval()
    max_len = cfg["eval_max_len"] if max_len is None else max_len
    refs, preds = [], []
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = greedy_decode(model, b["src"], b["src_mask"], cfg, tokenizer, max_len)
        for r, p in zip(b["tgt"].tolist(), ids.tolist()):
            if cfg["tokenization"] == "bytes":
                refs.append(bytes_to_text([x for x in r if x != 0]))
                preds.append(bytes_to_text(p))
            else:
                refs.append(bpe_decode(tokenizer, r))
                preds.append(bpe_decode(tokenizer, p))
    return utils.evaluate_texts(preds, refs, tokenized=(cfg["tokenization"] == "bpe"))


def train_one_config(name, cfg, tokenizer, smoke=False, use_wandb=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"\n=== {name}: attention={cfg['attention']} norm={cfg['norm']} "
          f"rope={cfg['use_rope']} tokenization={cfg['tokenization']} | {device} ===")

    tr, va, te = make_loaders(cfg, tokenizer)
    model = build_model(cfg, cfg["tgt_vocab"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    if use_wandb:
        import wandb
        wandb.init(project="anlp-assignment-1", name=name, config=cfg)

    hist, steps_done, epoch_times = [], 0, []
    last_loss = float("nan")
    n_epochs = 1 if smoke else cfg["epochs"]
    for ep in range(n_epochs):
        model.train()
        t0 = time.perf_counter()
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            last_loss = train_step(model, opt, b, cfg)
            steps_done += 1
            if steps_done % cfg["log_every"] == 0 or smoke:
                hist.append((steps_done, last_loss))
                if use_wandb:
                    wandb.log({"train/loss": last_loss, "step": steps_done})
            if smoke:
                break
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"epoch {ep + 1}: loss {last_loss:.3f} | {dt:.0f}s")

    metrics = evaluate(model, te, cfg, tokenizer,
                       max_len=64 if smoke else None,
                       max_batches=2 if smoke else None)
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    seconds_per_step = sum(epoch_times) / max(steps_done, 1)
    result = {"metrics": metrics, "train_time_s": sum(epoch_times),
              "seconds_per_step": seconds_per_step, "peak_mem_gb": peak_mem}
    print(f"{name} results: {result}")

    if use_wandb:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.log({"eval/seconds_per_step": seconds_per_step,
                   "eval/peak_mem_gb": peak_mem})
        wandb.finish()

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{name}.pt"))
    return result, hist


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all", help="e.g. C1,C2 or all")
    ap.add_argument("--smoke", action="store_true", help="1 batch, 1 epoch, no wandb")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    names = list(CONFIGS) if args.configs == "all" else args.configs.split(",")

    tokenizer = build_bpe_tokenizer(
        os.path.join(CONFIGS["C1"]["data_dir"], CONFIGS["C1"]["plain_file"]),
        CONFIGS["C1"]["tgt_vocab"],
        cache_path=os.path.join(OUT_DIR, f"bpe_{CONFIGS['C1']['tgt_vocab']}.json"))

    results, histories = {}, {}
    results_path = os.path.join(OUT_DIR, "ablation_results.json")
    if os.path.exists(results_path):
        import json as _json
        with open(results_path) as f:
            results = _json.load(f)
    for name in names:
        results[name], hist = train_one_config(name, CONFIGS[name], tokenizer,
                                               smoke=args.smoke,
                                               use_wandb=(not args.no_wandb and not args.smoke))
        histories[name] = hist

    utils.save_json(results, results_path)
    utils.save_json(histories, os.path.join(OUT_DIR, "histories.json"))
    utils.plot_loss_curves(histories, os.path.join(OUT_DIR, "loss_curves.png"))
    metric_maps = {k: v["metrics"] for k, v in results.items()}
    utils.plot_metrics_bar(metric_maps, os.path.join(OUT_DIR, "metrics_bar.png"))
    print("\n" + utils.markdown_table(metric_maps))


if __name__ == "__main__":
    main()
