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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_experiments as R
from src.data.claim_profiler import ClaimProfiler, significance
from src.data.loaders import load_liar2, load_xfact
from src.data.tokenizer import SimpleTokenizer
from src.evaluation.metrics import compute_classification_metrics
from src.models.ablation_study import NaiveConcatFusionNet, Pure1DCNNNet, PureSymbolicMLP
from src.models.me_fusion import MorphoEvidentialNet, MorphoTextDataset

MAX_LEN, BATCH_SIZE, LR, WEIGHT_DECAY = 64, 32, 1e-3, 1e-4
MIN_STEPS, MIN_EPOCHS, MAX_EPOCHS = 400, 5, 40
SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000]


def train_budgeted(model, seqs, ling, labels, seed):
    """Train with a fixed optimisation budget: >= MIN_STEPS updates, >= MIN_EPOCHS passes."""
    R.set_seed(seed)
    model = model.to(R.DEVICE)
    drop_last = len(labels) > BATCH_SIZE
    loader = DataLoader(MorphoTextDataset(seqs, ling, labels), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=drop_last, generator=torch.Generator().manual_seed(seed))
    steps_per_epoch = max(len(loader), 1)
    epochs = int(min(MAX_EPOCHS, max(MIN_EPOCHS, math.ceil(MIN_STEPS / steps_per_epoch))))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for s, l, y in loader:
            opt.zero_grad()
            logits, _ = model(s.to(R.DEVICE), l.to(R.DEVICE))
            crit(logits, y.to(R.DEVICE)).backward()
            opt.step()
    return model, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["xfact_id", "liar2", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 100])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/claims")
    args = ap.parse_args()
    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)

    jobs = []
    if args.dataset in ("xfact_id", "both"):
        jobs.append(("xfact_id", load_xfact("id"), "id", ["test", "ood"]))
    if args.dataset in ("liar2", "both"):
        jobs.append(("liar2", load_liar2(), "en", ["test"]))

    for name, data, lang, eval_splits in jobs:
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_path = out_dir / "runs.jsonl"
        profiler = ClaimProfiler(lang)
        feats = {k: profiler.profile(v["Text"]) for k, v in data.items()}
        y = {k: v["Label"].values.astype(np.float32) for k, v in data.items()}
        texts = {k: v["Text"].tolist() for k, v in data.items()}
        n_train = len(y["train"])
        print(f"\n=== {name} ({lang}): " + ", ".join(f"{k}={len(v)}" for k, v in y.items()), flush=True)

        sig = significance(feats["train"], y["train"])
        sig.to_csv(out_dir / "feature_significance.csv", index=False)
        print(sig[["Feature", "cohens_d", "p_holm", "zero_rate"]].head(5).round(3).to_string(index=False), flush=True)

        sizes = [s for s in SIZES if s < n_train] + [n_train]
        for size in sizes:
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                idx = np.arange(n_train) if size >= n_train else rng.choice(n_train, size, replace=False)
                tr_text = [texts["train"][i] for i in idx]
                tr_y = y["train"][idx]

                scaler = StandardScaler().fit(feats["train"].values[idx])
                ling = {k: scaler.transform(v.values).astype(np.float32) for k, v in feats.items()}
                tok = SimpleTokenizer(max_vocab=30000, max_len=MAX_LEN, min_freq=1).fit(tr_text)
                seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}
                vocab, n_ling = len(tok.word2idx), ling["train"].shape[1]

                def log(model_name, probs, extra=None):
                    for split in eval_splits:
                        p = probs[split]
                        m = compute_classification_metrics(y[split], (p >= 0.5).astype(int), y_prob=p, n_bins=15)
                        row = {"dataset": name, "language": lang, "model": model_name, "train_size": int(size),
                               "seed": seed, "split": split, **m, **(extra or {})}
                        with open(runs_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                    return compute_classification_metrics(y[eval_splits[0]], (probs[eval_splits[0]] >= 0.5).astype(int))["Macro_F1"]

                t0 = time.time()
                tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                Xtr = tf.fit_transform(tr_text)
                lr = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, tr_y)
                f_tfidf = log("TF-IDF + LR", {s: lr.predict_proba(tf.transform(texts[s]))[:, 1] for s in eval_splits})

                slr = LogisticRegression(max_iter=2000).fit(ling["train"][idx], tr_y)
                log("Symbolic LR", {s: slr.predict_proba(ling[s])[:, 1] for s in eval_splits})

                scores = {}
                for mname, build in [
                    ("Pure Symbolic MLP", lambda: PureSymbolicMLP(ling_dim=n_ling)),
                    ("Pure 1D-CNN", lambda: Pure1DCNNNet(vocab_size=vocab)),
                    ("Naive Concatenation", lambda: NaiveConcatFusionNet(vocab_size=vocab, ling_dim=n_ling)),
                    ("ME-Net (Full)", lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=n_ling)),
                ]:
                    model, epochs = train_budgeted(build(), seqs["train"][idx], ling["train"][idx], tr_y, seed)
                    probs = {s: R.predict(model, seqs[s], ling[s]) for s in eval_splits}
                    scores[mname] = log(mname, probs, {"epochs": epochs})
                print(f"  size={size:<6d} seed={seed:<4d} tfidf={f_tfidf:.4f} "
                      + " ".join(f"{a}={scores[k]:.4f}" for k, a in
                                 [("Pure Symbolic MLP", "sym"), ("Pure 1D-CNN", "cnn"),
                                  ("Naive Concatenation", "cat"), ("ME-Net (Full)", "menet")])
                      + f"  ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
