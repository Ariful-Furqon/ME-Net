"""Is routing capacity enough, or must the gate be trained on variable reliability?

The scalar convex gate restores per-document routing, but whether a run learns to
route is seed-dependent: at mask level 0.5 on IPH, 7 of 10 seeds drove the text
weight to ~0 and gained +0.12 Macro-F1 over the concat baseline, while 3 stayed
locked high (0.52-0.88) and lost. Two mechanisms could explain the failures:
sigmoid saturation locking the gate onto the stronger stream early, and the fact
that the gate is only ever trained on clean text, so it is never given a reason
to learn what unreliable input looks like.

This script tests the second. Training applies random token masking to the text
stream only; the symbolic features are computed from the raw text and stay clean,
exactly as at test time.

PRE-REGISTERED, fixed before the first run:

  Routing criterion  a seed "routes" if gate_mean at mask level 0.50 is < 0.20.
                     This is the same threshold used in the earlier post-hoc
                     split, fixed here in advance rather than chosen after.
  H1  augmented training raises the number of routing seeds above the observed
      baseline of 7/10 for the scalar gate.
  H2  augmented training raises mean Macro-F1 at mask level 0.50 for the scalar
      gate by more than one baseline standard deviation (baseline 0.5917,
      sd 0.1310, so the bar is > 0.7227).

Two concat variants are also trained with augmentation as controls. If
augmentation lifts every variant equally it is ordinary data augmentation and
says nothing about routing; the scalar gate must gain *more* for H1/H2 to mean
anything.

Seeds are ten that no earlier run used, so this is a fresh test rather than a
re-reading of the data that generated the hypothesis.
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

import scripts.run_experiments as R  # noqa: E402
import scripts.run_variants as V  # noqa: E402
from scripts.run_degradation import degrade, PAD, UNK  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.models.fusion_variants import FeatureConditioner, MENetV2  # noqa: E402
from src.data.tokenizer import SimpleTokenizer  # noqa: E402

ROUTING_THRESHOLD = 0.20
H2_BAR = 0.7227
BASELINE_ROUTING = 7

# name, gate_mode, gate_on_features, augment
CONDITIONS = [
    ("scalar, clean train",   "convex_scalar", True,  False),
    ("scalar, augmented",     "convex_scalar", True,  True),
    ("ME-Net v1, augmented",  "complementary", True,  True),
    ("concat, augmented",     "none",          False, True),
]


def augment_batch(seqs, gen, max_p=0.75):
    """Mask a random fraction of each example's real tokens, fraction ~ U(0, max_p)."""
    out = seqs.clone()
    n_real = (out != PAD).sum(dim=1)
    p = torch.rand(len(out), generator=gen, device=out.device) * max_p
    keep = torch.rand(out.shape, generator=gen, device=out.device)
    # mask position j of row i when it holds a real token and its draw falls under p_i
    mask = (out != PAD) & (keep < p.unsqueeze(1))
    out[mask] = UNK
    _ = n_real
    return out


def train(model, seqs, ling, labels, seed, augment):
    R.set_seed(seed)
    model = model.to(R.DEVICE)
    drop_last = len(labels) > V.BATCH_SIZE
    loader = DataLoader(R.MorphoTextDataset(seqs, ling, labels), batch_size=V.BATCH_SIZE,
                        shuffle=True, drop_last=drop_last,
                        generator=torch.Generator().manual_seed(seed))
    epochs = int(min(V.MAX_EPOCHS, max(V.MIN_EPOCHS,
                                       math.ceil(V.MIN_STEPS / max(len(loader), 1)))))
    opt = torch.optim.AdamW(model.parameters(), lr=V.LR, weight_decay=V.WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device=R.DEVICE).manual_seed(seed + 9973)
    for _ in range(epochs):
        model.train()
        for s, l, y in loader:
            s = s.to(R.DEVICE)
            if augment:
                s = augment_batch(s, gen)
            opt.zero_grad()
            logits, _ = model(s, l.to(R.DEVICE))
            crit(logits, y.to(R.DEVICE)).backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iph", "liar2", "xfact_id"], default="iph")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[201, 202, 203, 204, 205, 206, 207, 208, 209, 210])
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/augment")
    args = ap.parse_args()

    R.DEVICE = torch.device(args.device)
    print(f"device: {R.DEVICE}", flush=True)
    print(f"pre-registered: routes if gate_mean@0.5 < {ROUTING_THRESHOLD}; "
          f"H1 routing seeds > {BASELINE_ROUTING}/10; H2 F1@0.5 > {H2_BAR}\n", flush=True)

    splits, feats, max_len = V.load_dataset(args.dataset)
    texts = {k: v["Text"].tolist() for k, v in splits.items()}
    y = {k: v["Label"].values.astype(np.float32) for k, v in splits.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"{args.dataset}.jsonl"

    cond = FeatureConditioner("rankgauss").fit(feats["train"])
    ling = {k: cond.transform(v) for k, v in feats.items()}
    tok = SimpleTokenizer(max_vocab=30000, max_len=max_len, min_freq=1).fit(texts["train"])
    seqs = {k: tok.texts_to_sequences(v) for k, v in texts.items()}

    rows = []
    for name, gate_mode, gate_feats, aug in CONDITIONS:
        t0 = time.time()
        for seed in args.seeds:
            model = MENetV2(vocab_size=len(tok.word2idx),
                            ling_feature_dim=ling["train"].shape[1],
                            gate_mode=gate_mode, gate_on_features=gate_feats)
            model = train(model, seqs["train"], ling["train"], y["train"], seed, aug)
            for p in args.levels:
                d = degrade(seqs["dev"], "mask", p, seed)
                prob = R.predict(model, d, ling["dev"])
                m = compute_classification_metrics(y["dev"], (prob >= 0.5).astype(int),
                                                   y_prob=prob, n_bins=15)
                row = {"dataset": args.dataset, "variant": name, "augment": aug,
                       "level": p, "seed": seed, **m,
                       **V.gate_stats(model, d, ling["dev"])}
                rows.append(row)
                with open(runs_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        print(f"  {name:22s} done [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.dataset}_raw.csv", index=False)

    print("\n=== Macro-F1 by mask level ===")
    print(df.pivot_table(index="variant", columns="level",
                         values="Macro_F1", aggfunc="mean").round(4).to_string())
    print("\n=== gate_mean by mask level ===")
    if "gate_mean" in df:
        print(df.pivot_table(index="variant", columns="level",
                             values="gate_mean", aggfunc="mean").round(4).to_string())

    print("\n=== pre-registered tests ===")
    at50 = df[df.level == 0.5]
    for name, *_ in CONDITIONS:
        s = at50[at50.variant == name]
        if "gate_mean" not in s or s["gate_mean"].isna().all():
            n_route = float("nan")
        else:
            n_route = int((s["gate_mean"] < ROUTING_THRESHOLD).sum())
        print(f"  {name:22s} routing seeds={n_route}/{len(s)}   "
              f"F1@0.5={s.Macro_F1.mean():.4f}+-{s.Macro_F1.std():.4f}")

    aug_s = at50[at50.variant == "scalar, augmented"]
    n_route = int((aug_s["gate_mean"] < ROUTING_THRESHOLD).sum())
    f1 = aug_s.Macro_F1.mean()
    print(f"\n  H1 (routing seeds > {BASELINE_ROUTING}/10): "
          f"{n_route}/10 -> {'SUPPORTED' if n_route > BASELINE_ROUTING else 'NOT supported'}")
    print(f"  H2 (F1@0.5 > {H2_BAR}): "
          f"{f1:.4f} -> {'SUPPORTED' if f1 > H2_BAR else 'NOT supported'}")


if __name__ == "__main__":
    main()
