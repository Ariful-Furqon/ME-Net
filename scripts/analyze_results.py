import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

MODEL_ORDER = [
    "Length-only LR", "Symbolic LR (13 feats)", "Pure Symbolic MLP", "TF-IDF + LR", "IndoBERT-base (fine-tuned)",
    "Pure 1D-CNN (w/o Symbolic)", "Naive Concatenation (w/o Gate)", "ME-Net w/o Evidentiality Priors",
    "ME-Net w/o Morphological Priors", "ME-Net w/o Length Features", "ME-Net (Full)",
]
NEURAL_VARIANTS = MODEL_ORDER[5:10]
FEATURE_LABELS = {
    "word_count": "Word count", "char_count": "Character count", "avg_word_length": "Avg. word length",
    "type_token_ratio": "Type-token ratio", "hapax_legomena_ratio": "Hapax legomena ratio",
    "passive_prefix_rate": "Passive prefix (di-/ter-)", "active_prefix_rate": "Active prefix (meN-/ber-)",
    "nominal_ke_an_rate": "Nominal circumfix ke-an", "discourse_particle_rate": "Discourse particles",
    "deceptive_evidential_rate": "Rumour evidentials", "institutional_attribution_rate": "Institutional attribution",
    "sensational_marker_rate": "Sensational markers", "attribution_to_rumor_ratio": "Attribution/rumour ratio",
}


def load_runs(results: Path) -> pd.DataFrame:
    rows = []
    for f in list(results.glob("*/runs.jsonl")) + list(results.glob("*/indobert_runs.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def fmt(mean, std, n, bold=False, digits=3):
    s = f"{mean:.{digits}f}" if n <= 1 or np.isnan(std) else f"{mean:.{digits}f}\\,{{\\scriptsize$\\pm${std:.{digits}f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def wrap_tabular(colspec: str, header: str, body: str) -> str:
    return "\n".join(["\\begin{tabular}{" + colspec + "}", "\\toprule", header, "\\midrule", body,
                      "\\bottomrule", "\\end{tabular}"])


def main_table(runs: pd.DataFrame, setting: str) -> str:
    metrics = [("Macro_F1", False), ("ROC_AUC", False), ("ECE", True)]
    d = runs[runs.setting == setting]
    agg = d.groupby(["model", "split"])[[m for m, _ in metrics]].agg(["mean", "std", "count"])
    lines = []
    best = {}
    for split in ["test", "rahutomo"]:
        for m, lower in metrics:
            vals = {mod: agg.loc[(mod, split), (m, "mean")] for mod in MODEL_ORDER if (mod, split) in agg.index}
            best[(split, m)] = (min if lower else max)(vals, key=vals.get)
    for i, mod in enumerate(MODEL_ORDER):
        if (mod, "test") not in agg.index:
            continue
        if i in (3, 5, 10):
            lines.append("\\midrule")
        cells = [mod.replace("&", "\\&")]
        for split in ["test", "rahutomo"]:
            for m, _ in metrics:
                r = agg.loc[(mod, split), m]
                cells.append(fmt(r["mean"], r["std"], r["count"], bold=best[(split, m)] == mod))
        n = int(agg.loc[(mod, "test"), ("Macro_F1", "count")])
        cells.insert(1, str(n))
        lines.append(" & ".join(cells) + " \\\\")
    header = ("& & \\multicolumn{3}{c}{\\iph{} test (in-domain)} & \\multicolumn{3}{c}{Rahutomo-600 (cross-dataset)} \\\\\n"
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
              "Model & $n$ & Macro-F1 & ROC-AUC & ECE & Macro-F1 & ROC-AUC & ECE \\\\")
    return wrap_tabular("lccccccc", header, "\n".join(lines))


def significance_table(runs: pd.DataFrame) -> tuple[str, dict]:
    out, lines = {}, []
    for setting in ["original", "normalized"]:
        for split in ["test", "rahutomo"]:
            d = runs[(runs.setting == setting) & (runs.split == split)]
            full = d[d.model == "ME-Net (Full)"].set_index("seed")["Macro_F1"]
            tests = []
            for v in NEURAL_VARIANTS:
                other = d[d.model == v].set_index("seed")["Macro_F1"].reindex(full.index)
                diff = (full - other).values
                p_t = stats.ttest_rel(full.values, other.values).pvalue
                p_w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                tests.append((v, diff.mean(), diff.std(ddof=1), p_t, p_w, int((diff > 0).sum()), len(diff)))
            p_holm = multipletests([t[4] for t in tests], method="holm")[1]
            for (v, md, sd, p_t, p_w, wins, n), ph in zip(tests, p_holm):
                out[f"{setting}|{split}|{v}"] = {"delta_f1": md, "sd": sd, "p_ttest": p_t, "p_wilcoxon": p_w, "p_holm": ph, "wins": wins, "n": n}
    for v in NEURAL_VARIANTS:
        cells = [v]
        for setting in ["original", "normalized"]:
            for split in ["test", "rahutomo"]:
                r = out[f"{setting}|{split}|{v}"]
                star = "$^{*}$" if r["p_holm"] < 0.05 else ""
                cells.append(f"{100*r['delta_f1']:+.2f}{star} ({r['wins']}/{r['n']})")
        lines.append(" & ".join(cells) + " \\\\")
    header = ("& \\multicolumn{2}{c}{Original text} & \\multicolumn{2}{c}{Normalised text} \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
              "Comparison (\\menet{} $-$ variant) & in-domain & cross-dataset & in-domain & cross-dataset \\\\")
    return wrap_tabular("lcccc", header, "\n".join(lines)), out


def feature_table(results: Path) -> str:
    sig = pd.read_csv(results / "original" / "feature_significance.csv")
    sig = sig.reindex(sig["cohens_d"].abs().sort_values(ascending=False).index)
    lines = []
    for _, r in sig.iterrows():
        p = r["p_holm"]
        p_str = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"{FEATURE_LABELS[r['Feature']]} & {r['Hoax_Mean']:.3f} & {r['Valid_Mean']:.3f} & "
                     f"{r['cohens_d']:+.3f} & [{r['d_ci_lower']:+.3f}, {r['d_ci_upper']:+.3f}] & {p_str} \\\\")
    header = "Feature & Hoax & Valid & $d$ & 95\\% CI & $p_{\\text{Holm}}$ \\\\"
    return wrap_tabular("lrrrcr", header, "\n".join(lines))


def figures(results: Path, fig_dir: Path, runs: pd.DataFrame):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})

    # Effect sizes with 95% CI.
    sig = pd.read_csv(results / "original" / "feature_significance.csv").sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    y = np.arange(len(sig))
    ax.errorbar(sig["cohens_d"], y, xerr=[sig["cohens_d"] - sig["d_ci_lower"], sig["d_ci_upper"] - sig["cohens_d"]],
                fmt="o", color="#2b5d8a", ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y, [FEATURE_LABELS[f] for f in sig["Feature"]])
    ax.set_xlabel("Cohen's $d$ (hoax $-$ valid)")
    fig.tight_layout()
    fig.savefig(fig_dir / "effect_sizes.pdf")
    plt.close(fig)

    # Gate distribution per class (first seed).
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=False)
    for ax, split, title in zip(axes, ["test", "rahutomo"], ["IPH test (in-domain)", "Rahutomo-600 (cross-dataset)"]):
        g = pd.read_csv(results / "original" / f"gate_samples_{split}.csv")
        bins = np.linspace(g["gate"].min(), g["gate"].max(), 40)
        for lab, name, col in [(0, "valid", "#2b5d8a"), (1, "hoax", "#c0392b")]:
            ax.hist(g.loc[g.label == lab, "gate"], bins=bins, alpha=0.6, density=True, label=name, color=col)
        ax.set_title(title)
        ax.set_xlabel("mean text gate $\\bar{g}$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_distribution.pdf")
    plt.close(fig)

    # In-domain vs cross-dataset Macro-F1 for both settings.
    models = [m for m in MODEL_ORDER if m in set(runs.model)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, split, title in zip(axes, ["test", "rahutomo"], ["IPH test (in-domain)", "Rahutomo-600 (cross-dataset)"]):
        x = np.arange(len(models))
        for k, (setting, col) in enumerate([("original", "#2b5d8a"), ("normalized", "#e08e0b")]):
            d = runs[(runs.split == split) & (runs.setting == setting)].groupby("model")["Macro_F1"].agg(["mean", "std"]).reindex(models)
            ax.bar(x + (k - 0.5) * 0.38, d["mean"], 0.38, yerr=d["std"].fillna(0), color=col, label=setting, capsize=1.5, error_kw={"lw": 0.7})
        ax.set_xticks(x, [m.replace(" (fine-tuned)", "").replace("(w/o Symbolic)", "").replace("Naive Concatenation (w/o Gate)", "Naive concat")
                          .replace("ME-Net w/o ", "w/o ").replace(" Priors", "").replace(" Features", "") for m in models], rotation=60, ha="right", fontsize=7)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "indomain_vs_crossdataset.pdf")
    plt.close(fig)


def write_tex(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "%\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    results, paper = Path(args.results), Path(args.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    runs = load_runs(results)
    for setting in ["original", "normalized"]:
        if (runs.setting == setting).any():
            write_tex(paper / "tables" / f"main_{setting}.tex", main_table(runs, setting))
    sig_tex, sig = significance_table(runs)
    write_tex(paper / "tables" / "significance.tex", sig_tex)
    write_tex(paper / "tables" / "features.tex", feature_table(results))
    figures(results, paper / "figures", runs)

    gate = pd.concat([pd.read_csv(results / s / "gate_stats.csv").assign(setting=s) for s in ["original", "normalized"]])
    summary = {
        "means": (lambda t: t.set_axis([c if isinstance(c, str) else "_".join(c) for c in t.columns], axis=1)
                  .to_dict(orient="records"))(
            runs.groupby(["setting", "split", "model"])[["Macro_F1", "ROC_AUC", "ECE", "Brier_Score", "Accuracy", "Hoax_Recall"]]
            .agg(["mean", "std"]).round(4).reset_index()),
        "significance": sig,
        "gate": gate.groupby(["setting", "split", "label"])["gate_mean"].agg(["mean", "std"]).round(4).reset_index().to_dict(orient="records"),
        "params": runs.groupby("model")["params"].first().dropna().astype(int).to_dict() if "params" in runs else {},
        "train_seconds": runs.groupby("model")["train_seconds"].mean().dropna().round(1).to_dict() if "train_seconds" in runs else {},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(runs.groupby(["setting", "split", "model"])[["Macro_F1", "ROC_AUC", "ECE"]].agg(["mean", "std"]).round(4).to_string())
    print(json.dumps(sig, indent=1))
    print(gate.groupby(["setting", "split", "label"])["gate_mean"].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
