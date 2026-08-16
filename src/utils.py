import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def _bits(text):
    return "".join(f"{b:08b}" for b in text.encode("utf-8"))


def bit_accuracy(pred, ref):
    """% exact bit matches; extra bits of the longer sequence count as mismatches."""
    p, r = _bits(pred), _bits(ref)
    n = max(len(p), len(r))
    return (sum(a == b for a, b in zip(p, r)) / n) if n else 1.0


def sequence_accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(refs)


def levenshtein(pred, ref):
    return edit_distance(pred, ref)


def bleu(preds, refs):
    smooth = SmoothingFunction().method1
    return corpus_bleu([[r] for r in refs], preds, smoothing_function=smooth)


def rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    keys = ("rouge1", "rouge2", "rougeL")
    scores = {k: [] for k in keys}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in keys:
            scores[k].append(s[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def evaluate_texts(preds, refs, tokenized=True):
    """Full metric dict for a config. tokenized=True adds BLEU/ROUGE (C1-C4 only)."""
    res = {
        "bit_accuracy": float(np.mean([bit_accuracy(p, r) for p, r in zip(preds, refs)])),
        "sequence_accuracy": sequence_accuracy(preds, refs),
        "levenshtein": float(np.mean([levenshtein(p, r) for p, r in zip(preds, refs)])),
    }
    if tokenized:
        res["bleu"] = bleu(preds, refs)
        res.update(rouge(preds, refs))
    return res


def plot_loss_curves(histories, out_path):
    """histories: {config_name: [(step, train_loss), ...]}."""
    plt.figure(figsize=(8, 5))
    for name, steps in histories.items():
        if not steps:
            continue
        xs, ys = zip(*steps)
        plt.plot(xs, ys, label=name)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_bar(results, out_path):
    """results: {config_name: metric_dict}. Grouped bar chart."""
    names = list(results)
    metrics = ["bit_accuracy", "sequence_accuracy", "bleu"] if "bleu" in results[names[0]] \
        else ["bit_accuracy", "sequence_accuracy"]
    x = np.arange(len(names))
    w = 0.28
    plt.figure(figsize=(8, 5))
    for i, m in enumerate(metrics):
        vals = [results[n].get(m, 0.0) for n in names]
        plt.bar(x + i * w, vals, w, label=m)
    plt.xticks(x + w, names)
    plt.ylabel("score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def markdown_table(results):
    """Compact markdown table for README/report. results: {name: metric_dict}."""
    metric_order = ["bit_accuracy", "sequence_accuracy", "levenshtein",
                    "bleu", "rouge1", "rouge2", "rougeL"]
    header = "| config | " + " | ".join(metric_order) + " |"
    sep = "|" + "---|" * (len(metric_order) + 1)
    rows = [header, sep]
    for name, r in results.items():
        vals = [f"{r.get(m, float('nan')):.4f}" for m in metric_order]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(rows)
