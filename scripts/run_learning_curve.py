"""Does the gate stay collapsed when collapsing is no longer free?

v1 found the gate collapses to the concatenation regime. Every fusion variant
tested there ends in `torch.cat`, and under concatenation a constant gate is
absorbed by the next linear layer -- W1 diag(g) is just another learnable matrix
-- so collapsing costs the model nothing. The convex modes added to
FlexibleFusion mix the streams by summation instead, which makes a constant gate
destroy information.

Two things are measured across training-set sizes, because IPH is saturated at
100% of the data and only the low-data regime leaves room for an architectural
effect:

  Macro_F1             -- does the variant help, and where
  gate_sd_across_docs  -- did the gate actually start routing per document
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

# name, conditioner, gate_mode, gate_on_features, gate_penalty
CONDITIONS = [
    ("concat (no gate)",      "rankgauss", "none",           False, 0.0),
    ("ME-Net v1 (concat)",    "rankgauss", "complementary",  True,  0.0),
    ("convex (per-dim)",      "rankgauss", "convex",         True,  0.0),
    ("convex (scalar)",       "rankgauss", "convex_scalar",  True,  0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iph", "liar2", "xfact_id"], default="iph")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/learning_curve")
    args = ap.parse_args()

    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)

    splits, feats, max_len = V.load_dataset(args.dataset)
    texts = {k: v["Text"].tolist() for k, v in splits.items()}
    y = {k: v["Label"].values.astype(np.float32) for k, v in splits.items()}
    n_train = len(y["train"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"{args.dataset}.jsonl"
    print(f"{args.dataset}: train={n_train}  dev={len(y['dev'])}\n", flush=True)

    summary = []
    for frac in args.fractions:
        size = n_train if frac >= 1.0 else max(64, int(round(frac * n_train)))
        print(f"=== train fraction {frac:.0%}  (n={size}) ===", flush=True)
        for name, cond_mode, gate_mode, gate_feats, pen in CONDITIONS:
            t0, rows = time.time(), []
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                idx = (np.arange(n_train) if size >= n_train
                       else rng.choice(n_train, size, replace=False))
                cond = FeatureConditioner(cond_mode).fit(feats["train"][idx])
                ling = {k: cond.transform(v) for k, v in feats.items()}
                tok = SimpleTokenizer(max_vocab=30000, max_len=max_len, min_freq=1)
                tok = tok.fit([texts["train"][i] for i in idx])
                seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}
                model = MENetV2(vocab_size=len(tok.word2idx),
                                ling_feature_dim=ling["train"].shape[1],
                                gate_mode=gate_mode, gate_on_features=gate_feats)
                model = V.train(model, seqs["train"][idx], ling["train"][idx],
                                y["train"][idx], seed, gate_penalty=pen)
                p = R.predict(model, seqs["dev"], ling["dev"])
                m = compute_classification_metrics(y["dev"], (p >= 0.5).astype(int),
                                                   y_prob=p, n_bins=15)
                row = {"dataset": args.dataset, "fraction": frac, "train_size": int(len(idx)),
                       "variant": name, "gate_mode": gate_mode, "seed": seed, "split": "dev",
                       **m, **V.gate_stats(model, seqs["dev"], ling["dev"])}
                rows.append(row)
                with open(runs_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            d = pd.DataFrame(rows)
            gsd = d["gate_sd_across_docs"].mean() if "gate_sd_across_docs" in d else float("nan")
            print(f"  {name:22s} F1={d.Macro_F1.mean():.4f}+-{d.Macro_F1.std():.4f}"
                  f"   gate_sd={gsd:.4f}   [{time.time()-t0:.0f}s]", flush=True)
            summary.append({"fraction": frac, "train_size": int(size), "variant": name,
                            "f1_mean": float(d.Macro_F1.mean()),
                            "f1_sd": float(d.Macro_F1.std()), "gate_sd": float(gsd)})
        print(flush=True)

    s = pd.DataFrame(summary)
    print("=== Macro-F1 by training fraction ===")
    print(s.pivot(index="variant", columns="fraction", values="f1_mean").round(4).to_string())
    print("\n=== gate_sd_across_docs (0 = collapsed to a constant gate) ===")
    print(s.pivot(index="variant", columns="fraction", values="gate_sd").round(4).to_string())
    s.to_csv(out_dir / f"{args.dataset}_summary.csv", index=False)


if __name__ == "__main__":
    main()
