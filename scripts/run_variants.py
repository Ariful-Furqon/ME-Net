"""Variant study for the two measured defects: gate collapse and heavy-tailed symbolic features.

Everything is scored on VALIDATION data only. The test split is not touched here, so that a variant can
be chosen without contaminating the final comparison.

Conditions crossed:
  feature conditioning: standard | log | rankgauss
  gate mode:            none (concat) | complementary | independent | residual
  gate input:           projections only | projections + raw standardised features
  gate variance penalty: 0 | lambda > 0

Usage:
    python scripts/run_variants.py --dataset iph --seeds 13 21 42 87 100
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_experiments as R
from src.data.claim_profiler import ClaimProfiler
from src.data.linguistic_profiler import IndonesianLinguisticProfiler
from src.data.loaders import load_iph, load_liar2, load_xfact
from src.data.tokenizer import SimpleTokenizer
from src.evaluation.metrics import compute_classification_metrics
from src.models.fusion_variants import FeatureConditioner, MENetV2

BATCH_SIZE, LR, WEIGHT_DECAY = 64, 1e-3, 1e-4
MIN_STEPS, MIN_EPOCHS, MAX_EPOCHS = 400, 5, 40


def train(model, seqs, ling, labels, seed, gate_penalty=0.0):
    R.set_seed(seed)
    model = model.to(R.DEVICE)
    drop_last = len(labels) > BATCH_SIZE
    loader = DataLoader(R.MorphoTextDataset(seqs, ling, labels), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=drop_last, generator=torch.Generator().manual_seed(seed))
    epochs = int(min(MAX_EPOCHS, max(MIN_EPOCHS, math.ceil(MIN_STEPS / max(len(loader), 1)))))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for s, l, y in loader:
            opt.zero_grad()
            logits, _ = model(s.to(R.DEVICE), l.to(R.DEVICE))
            loss = crit(logits, y.to(R.DEVICE))
            if gate_penalty:
                loss = loss + gate_penalty * model.gate_variance_penalty()
            loss.backward()
            opt.step()
    return model


def gate_stats(model, seqs, ling):
    """Mean and per-document spread of the gate on held-out data."""
    vals = []
    loader = DataLoader(R.MorphoTextDataset(seqs, ling, np.zeros(len(seqs))), batch_size=256, shuffle=False)
    model.eval()
    with torch.no_grad():
        for s, l, _ in loader:
            model(s.to(R.DEVICE), l.to(R.DEVICE))
            g = model.fusion.last_gate
            if g is not None:
                vals.append(g.mean(dim=-1).cpu())
    if not vals:
        return {}
    v = torch.cat(vals).numpy()
    return {"gate_mean": float(v.mean()), "gate_sd_across_docs": float(v.std()),
            "gate_min": float(v.min()), "gate_max": float(v.max())}


def load_dataset(name):
    if name == "iph":
        s = load_iph(dedup=True)
        s["dev"] = s.pop("val")  # every dataset here exposes train/dev/test
        prof = IndonesianLinguisticProfiler()
        feats = {k: prof.profile_dataframe(v[["Text"]]).values for k, v in s.items()}
        return s, feats, 256
    if name == "liar2":
        s = load_liar2()
        prof = ClaimProfiler("en")
        return s, {k: prof.profile(v["Text"]).values for k, v in s.items()}, 64
    s = load_xfact("id")
    prof = ClaimProfiler("id")
    return s, {k: prof.profile(v["Text"]).values for k, v in s.items()}, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iph", "liar2", "xfact_id"], default="iph")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 100])
    ap.add_argument("--train_size", type=int, default=0, help="0 = all; otherwise subsample the training split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/variants")
    args = ap.parse_args()
    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)

    splits, feats, max_len = load_dataset(args.dataset)
    texts = {k: v["Text"].tolist() for k, v in splits.items()}
    y = {k: v["Label"].values.astype(np.float32) for k, v in splits.items()}
    n_train = len(y["train"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.dataset + (f"_n{args.train_size}" if args.train_size else "")
    runs_path = out_dir / f"{tag}.jsonl"

    conditions = [
        ("concat baseline",           "standard",  "none",           False, 0.0),
        ("base ME-Net",               "standard",  "complementary",  False, 0.0),
        ("+ log features",            "log",       "complementary",  False, 0.0),
        ("+ rankgauss features",      "rankgauss", "complementary",  False, 0.0),
        ("+ gate sees features",      "rankgauss", "complementary",  True,  0.0),
        ("+ gate variance penalty",   "rankgauss", "complementary",  True,  0.1),
        ("independent gates",         "rankgauss", "independent",    True,  0.0),
        ("residual gating",           "rankgauss", "residual",       True,  0.0),
        ("concat + rankgauss",        "rankgauss", "none",           False, 0.0),
    ]

    for name, cond_mode, gate_mode, gate_feats, pen in conditions:
        t0 = time.time()
        rows = []
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            idx = np.arange(n_train) if not args.train_size or args.train_size >= n_train \
                else rng.choice(n_train, args.train_size, replace=False)
            cond = FeatureConditioner(cond_mode).fit(feats["train"][idx])
            ling = {k: cond.transform(v) for k, v in feats.items()}
            tok = SimpleTokenizer(max_vocab=30000, max_len=max_len, min_freq=1).fit([texts["train"][i] for i in idx])
            seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}
            model = MENetV2(vocab_size=len(tok.word2idx), ling_feature_dim=ling["train"].shape[1],
                            gate_mode=gate_mode, gate_on_features=gate_feats)
            model = train(model, seqs["train"][idx], ling["train"][idx], y["train"][idx], seed, gate_penalty=pen)
            p = R.predict(model, seqs["dev"], ling["dev"])
            m = compute_classification_metrics(y["dev"], (p >= 0.5).astype(int), y_prob=p, n_bins=15)
            row = {"dataset": args.dataset, "train_size": int(len(idx)), "variant": name,
                   "conditioner": cond_mode, "gate_mode": gate_mode, "gate_on_features": gate_feats,
                   "gate_penalty": pen, "seed": seed, "split": "dev", **m,
                   **gate_stats(model, seqs["dev"], ling["dev"])}
            rows.append(row)
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        d = pd.DataFrame(rows)
        gsd = d["gate_sd_across_docs"].mean() if "gate_sd_across_docs" in d else float("nan")
        print(f"  {name:26s} dev F1={d.Macro_F1.mean():.4f}+-{d.Macro_F1.std():.4f}  "
              f"gate_sd={gsd:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
