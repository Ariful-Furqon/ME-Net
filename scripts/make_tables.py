"""Build the paper's result tables from the raw run logs.

Every number in the manuscript should be traceable to a file on disk and to a
stated number of seeds, so each table prints its own provenance: the source file,
the seeds it aggregates and how many. Runs that were superseded (the 3-seed CPU
degradation sweeps, archived as *_3seed.jsonl) are deliberately not read.

Writes LaTeX to results/tables/*.tex and prints a readable version.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("results/tables")
VARIANT_ORDER = ["concat (no gate)", "ME-Net v1 (concat)", "convex (per-dim)", "convex (scalar)"]
AUG_ORDER = ["scalar, clean train", "concat, augmented",
             "ME-Net v1, augmented", "scalar, augmented"]


def load(path, **kw):
    p = Path(path)
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    df = pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p, **kw)
    return df


def provenance(df, path):
    seeds = sorted(int(s) for s in df["seed"].unique())
    return f"source: {path}; {len(seeds)} seeds {seeds}"


def emit(name, df, caption, label, float_fmt="%.4f"):
    OUT.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(float_format=float_fmt, escape=True, caption=caption, label=label,
                      column_format="l" + "r" * df.shape[1])
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(df.round(4).to_string())
    print(f"  -> {OUT / f'{name}.tex'}\n")


def pivot(df, values, index="variant", columns=None, order=None):
    t = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if order:
        t = t.reindex([v for v in order if v in t.index])
    return t


def table_gate_dispersion():
    print("=" * 78)
    print("TABLE 1  Gate dispersion across documents, by fusion form")
    print("  Tests the ordering predicted by Prop. collapse / Cor. tied / Cor. scalar:")
    print("  a mechanism is used only to the extent that abandoning it costs something.")
    frames = []
    for ds in ("iph", "xfact_id", "liar2"):
        path = f"results/learning_curve/{ds}.jsonl"
        d = load(path)
        if d is None:
            continue
        print(f"  {provenance(d, path)}")
        t = pivot(d, "gate_sd_across_docs", columns="fraction", order=VARIANT_ORDER)
        t.insert(0, "dataset", ds)
        frames.append(t)
    if not frames:
        return
    out = pd.concat(frames).set_index("dataset", append=True).swaplevel().sort_index()
    emit("t1_gate_dispersion", out,
         "Gate dispersion $\\mathrm{sd}(\\bar g)$ across documents, by training fraction. "
         "A value near zero means the gate has collapsed to a constant.",
         "tab:gate-dispersion")


def table_learning_curve():
    print("=" * 78)
    print("TABLE 2  Macro-F1 by training fraction")
    print("  Restoring routing capacity does not buy in-domain accuracy.")
    frames = []
    for ds in ("iph", "xfact_id", "liar2"):
        path = f"results/learning_curve/{ds}.jsonl"
        d = load(path)
        if d is None:
            continue
        t = pivot(d, "Macro_F1", columns="fraction", order=VARIANT_ORDER)
        t.insert(0, "dataset", ds)
        frames.append(t)
    if not frames:
        return
    out = pd.concat(frames).set_index("dataset", append=True).swaplevel().sort_index()
    emit("t2_learning_curve", out,
         "Macro-F1 on the development split by fraction of the training set.",
         "tab:learning-curve")


def table_degradation():
    print("=" * 78)
    print("TABLE 3  Test-time corruption of the text stream")
    print("  Only the token sequence is corrupted; the symbolic features are computed")
    print("  from the raw text and stay intact. gate_mean is the weight on the text stream.")
    for ds in ("iph", "liar2"):
        path = f"results/degradation/{ds}_raw.csv"
        d = load(path)
        if d is None:
            continue
        print(f"\n  --- {ds} --- {provenance(d, path)}")
        m = d[d["kind"].isin(["mask", "clean"])].copy()
        m.loc[m["kind"] == "clean", "level"] = 0.0
        f1 = pivot(m, "Macro_F1", columns="level", order=VARIANT_ORDER)
        gm = pivot(m, "gate_mean", columns="level", order=VARIANT_ORDER)
        print("  Macro-F1:")
        emit(f"t3_{ds}_mask_f1", f1,
             f"Macro-F1 under token masking ({ds}).", f"tab:mask-f1-{ds}")
        print("  gate_mean (weight on the text stream):")
        emit(f"t3_{ds}_mask_gate", gm,
             f"Mean gate value under token masking ({ds}). A gate that routes should "
             f"fall as the text becomes unreadable.", f"tab:mask-gate-{ds}")


def table_augmentation():
    print("=" * 78)
    print("TABLE 4  Where robustness actually comes from")
    print("  Pre-registered in scripts/run_augment.py before the first run.")
    print("  The 'concat, augmented' control is what makes this table readable.")
    path = "results/augment/iph_raw.csv"
    d = load(path)
    if d is None:
        return
    print(f"  {provenance(d, path)}")
    f1 = pivot(d, "Macro_F1", columns="level", order=AUG_ORDER)
    emit("t4_augment_f1", f1,
         "Macro-F1 under token masking when the text stream is corrupted during "
         "training as well as at test time. Every augmented variant converges to the "
         "same band; the gate is not what delivers robustness.",
         "tab:augment-f1")
    at50 = d[d["level"] == 0.5]
    summ = at50.groupby("variant").agg(
        F1_at_0p5=("Macro_F1", "mean"), sd=("Macro_F1", "std"),
        routing_seeds=("gate_mean", lambda s: int((s < 0.20).sum()) if s.notna().any() else -1),
        n=("Macro_F1", "size"))
    summ = summ.reindex([v for v in AUG_ORDER if v in summ.index])
    emit("t4_augment_routing", summ,
         "Macro-F1 at mask level 0.5 and the number of seeds meeting the "
         "pre-registered routing criterion ($\\bar g < 0.20$). $-1$ marks a variant "
         "with no gate.", "tab:augment-routing")


if __name__ == "__main__":
    table_gate_dispersion()
    table_learning_curve()
    table_degradation()
    table_augmentation()
    print("=" * 78)
    print(f"LaTeX written to {OUT}/")
