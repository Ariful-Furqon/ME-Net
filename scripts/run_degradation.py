"""Does a gate that actually routes buy robustness when one stream degrades?

Gating exists to lean on whichever stream is reliable for a given input. On clean
in-domain data both streams are always reliable, so even a perfect gate buys
nothing -- which is what the learning-curve sweep found (gate_sd rises 17x,
Macro-F1 does not move). This script tests the premise directly.

Models are trained on CLEAN data and evaluated on a progressively corrupted text
stream. Only the token sequence is corrupted; the symbolic features are computed
from the raw text and stay intact, so the CNN view degrades while Phi does not.
Degrading at test time only is the honest version -- real degradation is not in
your training set.

The decisive readout is gate_mean, not F1. In the convex modes the gate weights
the text stream, so a gate that genuinely routes should LOWER it as the text
becomes unreadable. A gate that is a learned constant cannot react at all.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_experiments as R  # noqa: E402
import scripts.run_variants as V  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.models.fusion_variants import FeatureConditioner, MENetV2  # noqa: E402
from src.data.tokenizer import SimpleTokenizer  # noqa: E402

PAD, UNK = 0, 1

CONDITIONS = [
    ("concat (no gate)",   "none",          False),
    ("ME-Net v1 (concat)", "complementary", True),
    ("convex (per-dim)",   "convex",        True),
    ("convex (scalar)",    "convex_scalar", True),
]


def degrade(seqs, kind, p, seed):
    """Corrupt the token stream. p=0 leaves it untouched, p=1 destroys it."""
    if p <= 0:
        return seqs
    rng = np.random.RandomState(seed)
    out = seqs.copy()
    for i in range(len(out)):
        row = out[i]
        n = int((row != PAD).sum())
        if n == 0:
            continue
        if kind == "mask":
            # unreadable words: real tokens become <unk>
            k = int(round(p * n))
            if k:
                row[rng.choice(n, k, replace=False)] = UNK
        elif kind == "truncate":
            keep = max(1, int(round((1 - p) * n)))
            row[keep:] = PAD
        elif kind == "shuffle":
            # destroy word order, keep the bag of words
            k = int(round(p * n))
            if k > 1:
                pos = rng.choice(n, k, replace=False)
                row[pos] = row[rng.permutation(pos)]
        else:
            raise ValueError(kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iph", "liar2", "xfact_id"], default="iph")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--kinds", nargs="+", default=["mask", "truncate", "shuffle"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/degradation")
    args = ap.parse_args()

    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)

    splits, feats, max_len = V.load_dataset(args.dataset)
    texts = {k: v["Text"].tolist() for k, v in splits.items()}
    y = {k: v["Label"].values.astype(np.float32) for k, v in splits.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"{args.dataset}.jsonl"
    print(f"{args.dataset}: train={len(y['train'])}  dev={len(y['dev'])}\n", flush=True)

    rows = []
    for name, gate_mode, gate_feats in CONDITIONS:
        t0 = time.time()
        for seed in args.seeds:
            cond = FeatureConditioner("rankgauss").fit(feats["train"])
            ling = {k: cond.transform(v) for k, v in feats.items()}
            tok = SimpleTokenizer(max_vocab=30000, max_len=max_len, min_freq=1).fit(texts["train"])
            seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}
            model = MENetV2(vocab_size=len(tok.word2idx),
                            ling_feature_dim=ling["train"].shape[1],
                            gate_mode=gate_mode, gate_on_features=gate_feats)
            model = V.train(model, seqs["train"], ling["train"], y["train"], seed)

            for kind in args.kinds:
                for p in args.levels:
                    if p == 0.0 and kind != args.kinds[0]:
                        continue          # the clean point is shared across kinds
                    d = degrade(seqs["dev"], kind, p, seed)
                    prob = R.predict(model, d, ling["dev"])
                    m = compute_classification_metrics(y["dev"], (prob >= 0.5).astype(int),
                                                       y_prob=prob, n_bins=15)
                    row = {"dataset": args.dataset, "variant": name, "gate_mode": gate_mode,
                           "kind": ("clean" if p == 0.0 else kind), "level": p, "seed": seed,
                           **m, **V.gate_stats(model, d, ling["dev"])}
                    rows.append(row)
                    with open(runs_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
        print(f"  trained+evaluated {name:22s} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.dataset}_raw.csv", index=False)

    for kind in args.kinds:
        sub = df[df["kind"].isin([kind, "clean"])]
        sub = sub.assign(level=np.where(sub["kind"] == "clean", 0.0, sub["level"]))
        print(f"\n=== {kind}: Macro-F1 by corruption level ===")
        print(sub.pivot_table(index="variant", columns="level",
                              values="Macro_F1", aggfunc="mean").round(4).to_string())
        if "gate_mean" in sub:
            print(f"--- {kind}: gate_mean (weight on the TEXT stream; "
                  f"should fall if the gate routes) ---")
            print(sub.pivot_table(index="variant", columns="level",
                                  values="gate_mean", aggfunc="mean").round(4).to_string())

    print("\n=== robustness: Macro-F1 retained at the harshest level, vs clean ===")
    clean = df[df["kind"] == "clean"].groupby("variant")["Macro_F1"].mean()
    worst = args.levels[-1]
    for kind in args.kinds:
        w = df[(df["kind"] == kind) & (df["level"] == worst)].groupby("variant")["Macro_F1"].mean()
        print(f"  {kind}:")
        for variant in clean.index:
            if variant in w:
                print(f"    {variant:22s} {clean[variant]:.4f} -> {w[variant]:.4f}"
                      f"   ({w[variant]-clean[variant]:+.4f})")


if __name__ == "__main__":
    main()
