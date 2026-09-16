import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_experiments as R
from src.data.linguistic_profiler import IndonesianLinguisticProfiler
from src.data.loaders import load_iph, load_rahutomo
from src.data.tokenizer import SimpleTokenizer
from src.evaluation.metrics import compute_classification_metrics
from src.models.ablation_study import Pure1DCNNNet
from src.models.me_fusion import MorphoEvidentialNet


def evaluate(model, seqs, ling, labels):
    p = R.predict(model, seqs, ling)
    return compute_classification_metrics(labels, (p >= 0.5).astype(int), y_prob=p, n_bins=15)


def fit_neural(kind, vocab, n_ling, tr_seq, tr_ling, tr_y, seed):
    build = (lambda: MorphoEvidentialNet(vocab_size=vocab, ling_feature_dim=n_ling)) if kind == "menet" \
        else (lambda: Pure1DCNNNet(vocab_size=vocab))
    return R.train_neural(build(), tr_seq, tr_ling, tr_y, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/diagnostics.json")
    args = ap.parse_args()
    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)

    splits = load_iph(dedup=True)
    ood = load_rahutomo()
    prof = IndonesianLinguisticProfiler()
    feats = {k: prof.profile_dataframe(v[["Text"]]) for k, v in
             {"train": splits["train"], "test": splits["test"], "rahutomo": ood}.items()}
    y = {k: v["Label"].values.astype(np.float32) for k, v in
         {"train": splits["train"], "test": splits["test"], "rahutomo": ood}.items()}
    texts = {"train": splits["train"]["Text"].tolist(), "test": splits["test"]["Text"].tolist(),
             "rahutomo": ood["Text"].tolist()}

    scaler = StandardScaler().fit(feats["train"].values)
    ling = {k: scaler.transform(v.values).astype(np.float32) for k, v in feats.items()}
    tok = SimpleTokenizer().fit(texts["train"])
    seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}
    vocab, n_ling = len(tok.word2idx), ling["train"].shape[1]
    out = {}

    # ---------- A. Learning curve ----------
    print("A. learning curve", flush=True)
    rows = []
    for frac in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            n = max(int(frac * len(y["train"])), 64)
            idx = rng.choice(len(y["train"]), n, replace=False)
            for kind in ["cnn", "menet"]:
                m = fit_neural(kind, vocab, n_ling, seqs["train"][idx], ling["train"][idx], y["train"][idx], seed)
                r = evaluate(m, seqs["test"], ling["test"], y["test"])
                rows.append({"frac": frac, "n": n, "model": kind, "seed": seed,
                             "Macro_F1": r["Macro_F1"], "ROC_AUC": r["ROC_AUC"]})
            print(f"  frac={frac} seed={seed} cnn={rows[-2]['Macro_F1']:.4f} menet={rows[-1]['Macro_F1']:.4f}", flush=True)
    out["learning_curve"] = rows

    # ---------- B. Length-matched subsample ----------
    print("B. length-matched subsample", flush=True)
    all_df = pd.concat([
        splits["train"].assign(split="train"), splits["test"].assign(split="test")
    ], ignore_index=True)
    wc = all_df["Text"].str.split().str.len().values
    bins = np.quantile(wc, np.linspace(0, 1, 21))
    binid = np.clip(np.digitize(wc, bins[1:-1]), 0, 19)
    lab = all_df["Label"].values
    keep = []
    rng = np.random.RandomState(0)
    for b in range(20):
        pos = np.where((binid == b) & (lab == 1))[0]
        neg = np.where((binid == b) & (lab == 0))[0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    keep = np.array(sorted(keep))
    sub = all_df.iloc[keep].reset_index(drop=True)
    sub_feats = prof.profile_dataframe(sub[["Text"]])
    sub_y = sub["Label"].values.astype(np.float32)
    tr_i, te_i = train_test_split(np.arange(len(sub)), test_size=0.15, random_state=0, stratify=sub_y)
    sub_scaler = StandardScaler().fit(sub_feats.values[tr_i])
    sub_ling = sub_scaler.transform(sub_feats.values).astype(np.float32)
    sub_tok = SimpleTokenizer().fit(sub["Text"].iloc[tr_i].tolist())
    sub_seq = sub_tok.texts_to_sequences(sub["Text"].tolist())
    lcols = list(sub_feats.columns)
    len_idx = [lcols.index("word_count"), lcols.index("char_count")]

    b_rows = []
    lr_len = LogisticRegression(max_iter=2000).fit(sub_ling[tr_i][:, len_idx], sub_y[tr_i])
    p = lr_len.predict_proba(sub_ling[te_i][:, len_idx])[:, 1]
    b_rows.append({"model": "Length-only LR", "seed": 0, **compute_classification_metrics(sub_y[te_i], (p >= .5).astype(int), p)})
    lr_sym = LogisticRegression(max_iter=2000).fit(sub_ling[tr_i], sub_y[tr_i])
    p = lr_sym.predict_proba(sub_ling[te_i])[:, 1]
    b_rows.append({"model": "Symbolic LR", "seed": 0, **compute_classification_metrics(sub_y[te_i], (p >= .5).astype(int), p)})
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    Xtr = tf.fit_transform(sub["Text"].iloc[tr_i])
    lr_tf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, sub_y[tr_i])
    p = lr_tf.predict_proba(tf.transform(sub["Text"].iloc[te_i]))[:, 1]
    b_rows.append({"model": "TF-IDF + LR", "seed": 0, **compute_classification_metrics(sub_y[te_i], (p >= .5).astype(int), p)})
    for seed in args.seeds:
        for kind, name in [("cnn", "Pure 1D-CNN"), ("menet", "ME-Net (Full)")]:
            m = fit_neural(kind, len(sub_tok.word2idx), sub_ling.shape[1], sub_seq[tr_i], sub_ling[tr_i], sub_y[tr_i], seed)
            b_rows.append({"model": name, "seed": seed, **evaluate(m, sub_seq[te_i], sub_ling[te_i], sub_y[te_i])})
    out["length_matched"] = {"n": int(len(sub)), "n_test": int(len(te_i)), "rows": b_rows}
    print("  ", {r["model"]: round(r["Macro_F1"], 4) for r in b_rows}, flush=True)

    # ---------- C. Rahutomo-600 cross-validation ----------
    print("C. Rahutomo-600 cross-validation", flush=True)
    r_y = y["rahutomo"]
    r_feats = feats["rahutomo"].values
    c_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(skf.split(r_feats, r_y)):
        sc = StandardScaler().fit(r_feats[tr])
        rl = sc.transform(r_feats).astype(np.float32)
        lr = LogisticRegression(max_iter=2000).fit(rl[tr], r_y[tr])
        p = lr.predict_proba(rl[te])[:, 1]
        c_rows.append({"model": "Symbolic LR", "fold": fold, **compute_classification_metrics(r_y[te], (p >= .5).astype(int), p)})
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = tf.fit_transform([texts["rahutomo"][i] for i in tr])
        lr2 = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, r_y[tr])
        p = lr2.predict_proba(tf.transform([texts["rahutomo"][i] for i in te]))[:, 1]
        c_rows.append({"model": "TF-IDF + LR", "fold": fold, **compute_classification_metrics(r_y[te], (p >= .5).astype(int), p)})
        rtok = SimpleTokenizer(min_freq=1).fit([texts["rahutomo"][i] for i in tr])
        rseq = rtok.texts_to_sequences(texts["rahutomo"])
        for kind, name in [("cnn", "Pure 1D-CNN"), ("menet", "ME-Net (Full)")]:
            m = fit_neural(kind, len(rtok.word2idx), rl.shape[1], rseq[tr], rl[tr], r_y[tr], args.seeds[0])
            c_rows.append({"model": name, "fold": fold, **evaluate(m, rseq[te], rl[te], r_y[te])})
    out["rahutomo_cv"] = c_rows
    print("  ", pd.DataFrame(c_rows).groupby("model")["Macro_F1"].mean().round(4).to_dict(), flush=True)

    # ---------- D. Domain-shift probe ----------
    print("D. domain shift", flush=True)
    dom_X = np.vstack([feats["train"].values, feats["rahutomo"].values])
    dom_y = np.r_[np.zeros(len(feats["train"])), np.ones(len(feats["rahutomo"]))]
    dsc = StandardScaler().fit(dom_X)
    tr, te = train_test_split(np.arange(len(dom_y)), test_size=0.3, random_state=0, stratify=dom_y)
    dlr = LogisticRegression(max_iter=2000).fit(dsc.transform(dom_X)[tr], dom_y[tr])
    dp = dlr.predict_proba(dsc.transform(dom_X)[te])[:, 1]
    dom = {"symbolic_domain_auc": compute_classification_metrics(dom_y[te], (dp >= .5).astype(int), dp)["ROC_AUC"]}
    shift = {}
    for c in feats["train"].columns:
        a, b = feats["train"][c].values, feats["rahutomo"][c].values
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
        shift[c] = {"iph_mean": float(a.mean()), "rahutomo_mean": float(b.mean()), "std_diff": float((b.mean() - a.mean()) / s)}
    dom["per_feature"] = shift
    # class balance / prior shift and vocabulary overlap
    dom["iph_hoax_rate"] = float(y["train"].mean())
    dom["rahutomo_hoax_rate"] = float(y["rahutomo"].mean())
    riph = set(tok.word2idx)
    rvocab = SimpleTokenizer(min_freq=1).fit(texts["rahutomo"]).word2idx
    dom["rahutomo_vocab_in_iph"] = float(np.mean([w in riph for w in rvocab]))
    out["domain_shift"] = dom
    print("  ", {k: v for k, v in dom.items() if k != "per_feature"}, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("written", args.out, flush=True)


if __name__ == "__main__":
    main()
