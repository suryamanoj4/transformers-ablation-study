#!/usr/bin/env python
"""Upload the trained model checkpoints in output/ to Hugging Face Hub.

Usage:
    export HF_TOKEN=hf_xxx                      # or pass --token
    .venv/bin/python tools/upload_hf.py --repo-id yourname/anlp-assignment-1
    .venv/bin/python tools/upload_hf.py --private   # or make it private

Uploads (from --out-dir):
    C1.pt ... C5.pt        model state dicts
    bpe_*.json             from-scratch byte-level BPE tokenizer caches
    ablation_results.json  test metrics / compute profile
    README.md              generated card (summary + configs + metrics)
"""
import argparse
import datetime
import json
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("pip install huggingface_hub  (and run: huggingface-cli login / export HF_TOKEN)")
    sys.exit(1)


def load_env_token():
    """HF token from environment or .env; never printed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line.startswith("HF_TOKEN") and "=" in line and "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def make_readme(out_dir):
    """Auto-generated model card: file list + metrics from results JSON."""
    info = sorted(f for f in os.listdir(out_dir) if f.endswith(".pt"))
    if not info:
        raise SystemExit(f"no .pt files found in {out_dir} (nothing to upload)")
    mtime = lambda f: datetime.datetime.fromtimestamp(
        os.path.getmtime(os.path.join(out_dir, f))).strftime("%Y-%m-%d %H:%M")
    frows = "| checkpoint | size | date |\n|---|---|---|\n" + "\n".join(
        f"| {f} | {os.path.getsize(os.path.join(out_dir, f))/1e6:.1f} MB | {mtime(f)} |"
        for f in info)

    metrics = ""
    rpath = os.path.join(out_dir, "ablation_results.json")
    if os.path.exists(rpath):
        d = json.load(open(rpath))
        cols = ["bit_accuracy", "sequence_accuracy", "levenshtein", "bleu", "params"]
        metrics = ("| config | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n")
        for name, r in d.items():
            mm = r.get("metrics", {})
            vals = [f"{mm.get(c, float('nan')):.4f}" if c in mm else f"{r.get(c, '')}"
                    for c in cols]
            metrics += f"| {name} | " + " | ".join(vals) + " |\n"

    return f"""---
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

{metrics}
## Files
{frows}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="suryamanojphy31/anlp-assignment-1",
                    help="HF repository to create/update")
    ap.add_argument("--out-dir", default="output", help="directory with checkpoints")
    ap.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN or .env)")
    ap.add_argument("--private", action="store_true", help="create a private repo")
    args = ap.parse_args()

    token = args.token or load_env_token()
    if not token:
        raise SystemExit("no HF token: export HF_TOKEN=... or pass --token")

    api = HfApi(token=token)
    try:
        api.repo_info(args.repo_id, token=token)
        print(f"repo exists: {args.repo_id}")
    except Exception:
        print(f"creating repo: {args.repo_id} (private={args.private})")
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    README = os.path.join(args.out_dir, "README.md")
    with open(README, "w", encoding="utf-8") as f:
        f.write(make_readme(args.out_dir))
    print(f"wrote {README}")

    files = sorted(
        f for f in os.listdir(args.out_dir)
        if f.endswith((".pt", ".json")) or f == "README.md")
    skip = {"histories.json", "train.log", "train2.log"}
    for name in files:
        if name in skip:
            continue
        path = os.path.join(args.out_dir, name)
        print(f"uploading {name} ...", end=" ", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=args.repo_id, token=token)
        print("ok")

    print(f"\nDone. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
