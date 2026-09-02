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

BASE = dict(d_model=256, d_ff=1024, n_heads=8, n_kv_heads=2, n_layers=3,
            dropout=0.1, max_len=1024, eval_max_len=512, patch_size=8,
            n_local_layers=2, tgt_vocab=4000, src_vocab_cipher=1500,
            batch_size=16, warmup_steps=1000, label_smoothing=0.1,
            amp=True, max_len_bytes=12288, eval_max_len_bytes=2048,
            max_len_patches=1024, local_window=512,
            epochs=20, log_every=1, patience=3, min_delta=0.003,
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


def build_model(cfg, tgt_vocab, src_vocab=None):
    model_cfg = dict(cfg, attention=cfg["attention"], norm=cfg["norm"],
                     use_rope=cfg["use_rope"])
    if cfg["tokenization"] == "bytes":
        mc = dict(model_cfg)
        mc["max_len"] = max(cfg["max_len"], cfg.get("max_len_patches", 1024))
        return BLTModel(mc, cfg["patch_size"], cfg["n_local_layers"])
    src_vocab = 256 if src_vocab is None else src_vocab
    return EncoderDecoder(model_cfg, src_vocab_size=src_vocab,
                          tgt_vocab_size=tgt_vocab)


def noam_lr(cfg, step):
    """Inverse-sqrt LR with linear warmup: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    ws = max(cfg["warmup_steps"], 1)
    return cfg["d_model"] ** -0.5 * min(step ** -0.5, step * ws ** -1.5)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batch pairs grouped by approximate sequence length (less padding waste).
    Batch order is re-shuffled every epoch (same generator -> reproducible)."""

    def __init__(self, ds, n, batch_size, generator, k=4):
        self.generator = generator
        lens = [ds[i]["src"].numel() + ds[i]["tgt"].numel() for i in range(n)]
        order = sorted(range(n), key=lambda i: lens[i])
        batches = []
        for s in range(0, n, batch_size * k):
            chunk = order[s:s + batch_size * k]
            perm = torch.randperm(len(chunk), generator=generator).tolist()
            chunk = [chunk[j] for j in perm]
            for t in range(0, len(chunk), batch_size):
                b = chunk[t:t + batch_size]
                if len(b) >= 4:
                    batches.append(b)
        self.batches = batches

    def __iter__(self):
        perm = torch.randperm(len(self.batches), generator=self.generator).tolist()
        for i in perm:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def make_loaders(cfg, src_tokenizer, tgt_tokenizer):
    mode = cfg["tokenization"]
    pad = 0 if mode == "bytes" else tgt_tokenizer.token_to_id("<pad>")
    max_len = cfg["max_len_bytes"] if mode == "bytes" else cfg["max_len"]
    ds = CipherDataset(os.path.join(cfg["data_dir"], cfg["cipher_file"]),
                       os.path.join(cfg["data_dir"], cfg["plain_file"]),
                       src_tokenizer, tgt_tokenizer, mode=mode, max_len=max_len)
    gen = torch.Generator().manual_seed(cfg["seed"])
    n = len(ds)
    n_tr = int(n * cfg["train_frac"])
    n_va = int(n * cfg["val_frac"])
    tr, va, te = random_split(ds, [n_tr, n_va, n - n_tr - n_va], gen)
    tr_loader = DataLoader(tr, batch_size=1, shuffle=False,
                           batch_sampler=BucketBatchSampler(tr, n_tr, cfg["batch_size"], gen),
                           collate_fn=lambda b: collate(b, pad_tgt_id=pad))
    mk = lambda d: DataLoader(d, batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=lambda b: collate(b, pad_tgt_id=pad))
    return tr_loader, mk(va), mk(te)


def _logits_and_labels(model, batch, cfg):
    src, tgt, sm, tm = batch["src"], batch["tgt"], batch["src_mask"], batch["tgt_mask"]
    if cfg["tokenization"] == "bytes":
        logits = model(src, tgt, sm)
        labels = tgt.clone().masked_fill(~tm[:, 0, 0], -100)
    else:
        logits = model(src, tgt[:, :-1], sm, tm[:, :, :, :-1])
        labels = tgt[:, 1:].masked_fill(~tm[:, 0, 0, 1:], -100)
    return logits, labels


def train_step(model, opt, batch, cfg, scaler=None):
    with torch.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
        logits, labels = _logits_and_labels(model, batch, cfg)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100,
                               label_smoothing=cfg["label_smoothing"])
    opt.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    return loss.item()


@torch.no_grad()
def val_loss(model, loader, cfg, device):
    model.eval()
    use_amp = torch.cuda.is_available()
    total, ntok = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits, labels = _logits_and_labels(model, b, cfg)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100,
                                   label_smoothing=cfg["label_smoothing"])
        n = (labels != -100).sum().item()
        total += loss.item() * n
        ntok += n
    model.train()
    return total / max(ntok, 1)


@torch.no_grad()
def greedy_decode(model, src, sm, cfg, tokenizer, max_len):
    """C1-C4: token loop. C5: patch loop (patch_size bytes per step)."""
    device = next(model.parameters()).device
    src = src.to(device)
    sm = sm.to(device)
    model.eval()
    use_amp = torch.cuda.is_available()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
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
    if max_len is None:
        max_len = (cfg["eval_max_len_bytes"] if cfg["tokenization"] == "bytes"
                   else cfg["eval_max_len"])
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


def _early_stop_update(best_val, best_epoch, no_improve, vloss, min_delta, ep):
    """returns (best_val, best_epoch, no_improve) after one epoch's val loss."""
    if vloss < best_val - min_delta:
        return vloss, ep + 1, 0
    return best_val, best_epoch, no_improve + 1


def train_one_config(name, cfg, tokenizer, cipher_tok=None, smoke=False, use_wandb=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"\n=== {name}: attention={cfg['attention']} norm={cfg['norm']} "
          f"rope={cfg['use_rope']} tokenization={cfg['tokenization']} | {device} ===")

    src_vocab = (cipher_tok.get_vocab_size() if cipher_tok is not None else 256)
    tr, va, te = make_loaders(cfg, cipher_tok, tokenizer)
    model = build_model(cfg, cfg["tgt_vocab"], src_vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=noam_lr(cfg, 1),
                            betas=(0.9, 0.98), eps=1e-9)
    scaler = (torch.amp.GradScaler(enabled=True)
              if (cfg["amp"] and torch.cuda.is_available()) else None)

    if use_wandb:
        import wandb
        wandb.init(project="anlp-assignment-1", name=name, config=cfg)

    hist, steps_done, epoch_times = [], 0, []
    last_loss, best_val = float("nan"), float("inf")
    snapshots = []
    best_epoch, no_improve, stop = 0, 0, False
    n_epochs = 1 if smoke else cfg["epochs"]
    for ep in range(n_epochs):
        model.train()
        t0 = time.perf_counter()
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            lr = noam_lr(cfg, steps_done + 1)
            for g in opt.param_groups:
                g["lr"] = lr
            last_loss = train_step(model, opt, b, cfg, scaler)
            steps_done += 1
            if steps_done % cfg["log_every"] == 0 or smoke:
                hist.append((steps_done, last_loss))
                if use_wandb:
                    wandb.log({"train/loss": last_loss, "train/lr": lr,
                               "step": steps_done})
            if smoke:
                break
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        vloss = val_loss(model, va, cfg, device)
        best_val, best_epoch, no_improve = _early_stop_update(
            best_val, best_epoch, no_improve, vloss, cfg["min_delta"], ep)
        snapshots.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if len(snapshots) > 5:
            snapshots.pop(0)
        if use_wandb:
            wandb.log({"val/loss": vloss, "epoch": ep + 1})
        print(f"epoch {ep + 1}: train {last_loss:.3f} | val {vloss:.3f} | {dt:.0f}s")
        if not smoke and no_improve >= cfg["patience"]:
            stop = True
            print(f"early stop: val loss flat for {cfg['patience']} epochs (best at epoch {best_epoch}, {best_val:.4f})")
            break

    # average the last 5 epoch snapshots, then evaluate that on the test set
    avg_state = {k: torch.stack([s[k] for s in snapshots]).mean(0)
                 for k in snapshots[0]}
    model.load_state_dict(avg_state)
    metrics = evaluate(model, te, cfg, tokenizer,
                       max_len=64 if smoke else None,
                       max_batches=2 if smoke else None)
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    seconds_per_step = sum(epoch_times) / max(steps_done, 1)
    result = {"metrics": metrics, "train_time_s": sum(epoch_times),
              "seconds_per_step": seconds_per_step, "peak_mem_gb": peak_mem,
              "best_val_loss": best_val, "best_val_epoch": best_epoch,
              "stopped_early": bool(stop)}
    print(f"{name} results: {result}")

    if use_wandb:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.log({"eval/seconds_per_step": seconds_per_step,
                   "eval/peak_mem_gb": peak_mem, "eval/best_val_loss": best_val})
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

    cipher_tok = build_bpe_tokenizer(
        os.path.join(CONFIGS["C1"]["data_dir"], CONFIGS["C1"]["cipher_file"]),
        CONFIGS["C1"]["src_vocab_cipher"],
        cache_path=os.path.join(OUT_DIR, f"bpe_cipher_{CONFIGS['C1']['src_vocab_cipher']}.json"))

    results, histories = {}, {}
    results_path = os.path.join(OUT_DIR, "ablation_results.json")
    if os.path.exists(results_path):
        import json as _json
        with open(results_path) as f:
            results = _json.load(f)
    for name in names:
        results[name], hist = train_one_config(name, CONFIGS[name], tokenizer,
                                               cipher_tok=cipher_tok,
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
