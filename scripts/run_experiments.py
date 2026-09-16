"""Main experiment driver for ME-Net.

For one input setting (original text or format-normalised text) it trains every model
variant over several seeds on the Indonesia Political Hoax (IPH) training split and
evaluates on the IPH test split (in-domain) and on Rahutomo-600 (cross-dataset).

Usage:
    python scripts/run_experiments.py --setting original --seeds 13 21 42 87 100
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.linguistic_profiler import IndonesianLinguisticProfiler
from src.data.loaders import load_iph, load_rahutomo, normalize_surface
from src.data.tokenizer import SimpleTokenizer
from src.evaluation.metrics import compute_classification_metrics
from src.models.ablation_study import NaiveConcatFusionNet, Pure1DCNNNet, PureSymbolicMLP
from src.models.me_fusion import MorphoEvidentialNet, MorphoTextDataset

EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY = 5, 64, 1e-3, 1e-4
DEVICE = torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, seqs, ling):
    model.eval()
    loader = DataLoader(MorphoTextDataset(seqs, ling, np.zeros(len(seqs))), batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for s, l, _ in loader:
            logits, _ = model(s.to(DEVICE), l.to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_neural(model, seqs, ling, labels, seed):
    set_seed(seed)
    model = model.to(DEVICE)
    loader = DataLoader(MorphoTextDataset(seqs, ling, labels), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train()
        for s, l, y in loader:
            opt.zero_grad()
            logits, _ = model(s.to(DEVICE), l.to(DEVICE))
            crit(logits, y.to(DEVICE)).backward()
            opt.step()
    return model


def gate_statistics(model, seqs, ling):
    """Mean text-stream gate value g per sample (g -> 1: text stream dominates, g -> 0: linguistic stream)."""
    captured = []
    hook = model.fusion.gate.register_forward_hook(lambda m, i, o: captured.append(o.mean(dim=-1).detach().cpu()))
    predict(model, seqs, ling)
    hook.remove()
    return torch.cat(captured).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["original", "normalized"], default="original")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 100])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global DEVICE
    DEVICE = torch.device(args.device)
    print(f"device: {DEVICE}", flush=True)

    out_dir = Path(args.out) / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_iph(dedup=True)
    ood = load_rahutomo()
    data = {"train": splits["train"], "val": splits["val"], "test": splits["test"], "rahutomo": ood}
    if args.setting == "normalized":
        for d in data.values():
            d["Text"] = d["Text"].map(normalize_surface)
    print({k: (len(v), int(v["Label"].sum())) for k, v in data.items()}, flush=True)

    profiler = IndonesianLinguisticProfiler()
    feats = {k: profiler.profile_dataframe(v[["Text"]]) for k, v in data.items()}
    ling_cols = list(feats["train"].columns)
    scaler = StandardScaler().fit(feats["train"].values)
    ling = {k: scaler.transform(v.values).astype(np.float32) for k, v in feats.items()}
    labels = {k: v["Label"].values.astype(np.float32) for k, v in data.items()}

    if args.setting == "original":
        sig = profiler.compute_linguistic_significance(data["train"], text_col="Text", label_col="Label")
        sig.to_csv(out_dir / "feature_significance.csv", index=False)

    tok = SimpleTokenizer(max_vocab=30000, max_len=256).fit(data["train"]["Text"].tolist())
    seqs = {k: tok.texts_to_sequences(v["Text"].tolist()) for k, v in data.items()}
    vocab = len(tok.word2idx)

    morph_idx = [i for i, c in enumerate(ling_cols) if any(k in c for k in ["prefix", "ke_an", "particle"])]
    evid_idx = [i for i, c in enumerate(ling_cols) if any(k in c for k in ["evidential", "attribution", "sensational", "ratio"])]
    length_idx = [ling_cols.index("word_count"), ling_cols.index("char_count")]
    n_ling = len(ling_cols)

    runs_path = out_dir / "runs.jsonl"
    preds_store = {}

    def record(name, seed, probs_by_split, extra=None):
        for split in ["test", "rahutomo"]:
            p = probs_by_split[split]
            m = compute_classification_metrics(labels[split], (p >= 0.5).astype(int), y_prob=p, n_bins=15)
            row = {"setting": args.setting, "model": name, "seed": seed, "split": split, **m, **(extra or {})}
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            preds_store[f"{name}|{seed}|{split}"] = p
        print(f"  {name:32s} seed={seed:<4d} test F1={compute_classification_metrics(labels['test'], (probs_by_split['test']>=.5).astype(int))['Macro_F1']:.4f}", flush=True)

    # Deterministic classical baselines (seed-independent).
    t0 = time.time()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    Xtr = tfidf.fit_transform(data["train"]["Text"])
    lr = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, labels["train"])
    record("TF-IDF + LR", 0, {s: lr.predict_proba(tfidf.transform(data[s]["Text"]))[:, 1] for s in ["test", "rahutomo"]})

    for name, idx in [("Length-only LR", length_idx), ("Symbolic LR (13 feats)", list(range(n_ling)))]:
        clf = LogisticRegression(max_iter=2000).fit(ling["train"][:, idx], labels["train"])
        record(name, 0, {s: clf.predict_proba(ling[s][:, idx])[:, 1] for s in ["test", "rahutomo"]})
    print(f"classical baselines done in {time.time()-t0:.0f}s", flush=True)

    variants = [
        ("Pure Symbolic MLP", lambda: PureSymbolicMLP(ling_dim=n_ling), list(range(n_ling))),
        ("Pure 1D-CNN (w/o Symbolic)", lambda: Pure1DCNNNet(vocab_size=vocab), list(range(n_ling))),
        ("Naive Concatenation (w/o Gate)", lambda: NaiveConcatFusionNet(vocab_size=vocab, ling_dim=n_ling), list(range(n_ling))),
        ("ME-Net w/o Evidentiality Priors", lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=len(morph_idx)), morph_idx),
        ("ME-Net w/o Morphological Priors", lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=len(evid_idx)), evid_idx),
        ("ME-Net w/o Length Features", lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=n_ling - 2), [i for i in range(n_ling) if i not in length_idx]),
        ("ME-Net (Full)", lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=n_ling), list(range(n_ling))),
    ]

    gate_rows = []
    for seed in args.seeds:
        for name, build, idx in variants:
            t0 = time.time()
            set_seed(seed)
            model = train_neural(build(), seqs["train"], ling["train"][:, idx], labels["train"], seed)
            probs = {s: predict(model, seqs[s], ling[s][:, idx]) for s in ["val", "test", "rahutomo"]}
            n_params = sum(p.numel() for p in model.parameters())
            record(name, seed, probs, {"params": n_params, "train_seconds": round(time.time() - t0, 1)})
            if name == "ME-Net (Full)":
                for s in ["test", "rahutomo"]:
                    g = gate_statistics(model, seqs[s], ling[s])
                    for lab in [0, 1]:
                        mask = labels[s] == lab
                        gate_rows.append({"seed": seed, "split": s, "label": lab, "gate_mean": float(g[mask].mean()), "gate_std": float(g[mask].std())})
                    if seed == args.seeds[0]:
                        pd.DataFrame({"gate": g, "label": labels[s], "prob": probs[s], **{c: feats[s][c].values for c in ling_cols}}).to_csv(out_dir / f"gate_samples_{s}.csv", index=False)

    pd.DataFrame(gate_rows).to_csv(out_dir / "gate_stats.csv", index=False)
    np.savez_compressed(out_dir / "predictions.npz", **{k.replace("|", "__"): v for k, v in preds_store.items()},
                        y_test=labels["test"], y_rahutomo=labels["rahutomo"])
    with open(out_dir / "data_summary.json", "w") as f:
        json.dump({k: {"n": len(v), "hoax": int(v["Label"].sum()), "median_words": float(v["Text"].str.split().str.len().median())}
                   for k, v in data.items()} | {"vocab": vocab, "ling_cols": ling_cols}, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
