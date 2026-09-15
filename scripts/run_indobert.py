"""Fine-tuned IndoBERT baseline (CPU-friendly settings) for the same splits used by run_experiments.py.

Usage:
    python scripts/run_indobert.py --setting original --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_iph, load_rahutomo, normalize_surface
from src.evaluation.metrics import compute_classification_metrics

MODEL_NAME = "indobenchmark/indobert-base-p1"


def encode(tok, texts, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def predict(model, ids, mask, batch_size=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            logits = model(input_ids=ids[i:i + batch_size], attention_mask=mask[i:i + batch_size]).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["original", "normalized"], default="original")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_iph(dedup=True)
    data = {"train": splits["train"], "test": splits["test"], "rahutomo": load_rahutomo()}
    if args.setting == "normalized":
        for d in data.values():
            d["Text"] = d["Text"].map(normalize_surface)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    enc = {k: encode(tok, v["Text"], args.max_len) for k, v in data.items()}
    y = {k: v["Label"].values for k, v in data.items()}

    loader = DataLoader(TensorDataset(*enc["train"], torch.tensor(y["train"], dtype=torch.long)),
                        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for ids, mask, labels in loader:
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {loss.item():.4f} elapsed {time.time()-t0:.0f}s", flush=True)

    out_dir = Path(args.out) / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for split in ["test", "rahutomo"]:
        p = predict(model, *enc[split])
        preds[split] = p
        m = compute_classification_metrics(y[split], (p >= 0.5).astype(int), y_prob=p, n_bins=15)
        row = {"setting": args.setting, "model": "IndoBERT-base (fine-tuned)", "seed": args.seed, "split": split, **m,
               "params": sum(q.numel() for q in model.parameters()), "train_seconds": round(time.time() - t0, 1)}
        with open(out_dir / "indobert_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
    np.savez_compressed(out_dir / f"indobert_preds_seed{args.seed}.npz", **preds)


if __name__ == "__main__":
    main()
