# ME-Net: Morpho-Evidential Gated Dual-Stream Network for Indonesian Hoax Detection

ME-Net combines a 1D-CNN over tokens with a 13-dimensional Indonesian linguistic profile
(surface statistics, lexicon-verified affix rates, rumour/attribution evidentials, sensational
markers), fused through a learned complementary gate.

> **Status:** work in progress. Code and paper draft are in place; the full multi-seed
> experiments have not been completed yet, so the paper has no final results.

## Layout

```
src/
  data/
    linguistic_profiler.py   # 13-feature Indonesian morpho-evidential profiler + significance tests
    affix_exceptions.py      # exception lists for the affix/clitic detector
    lexicon_root.txt         # 29,931 Indonesian root words (Sastrawi kata-dasar list)
    tokenizer.py             # word-level tokenizer
    loaders.py               # IPH (deduplicated splits), Rahutomo-600, format normalisation
  models/
    me_fusion.py             # ME-Net and GatedCrossModalFusion
    ablation_study.py        # ablation variants (naive concat, pure CNN, pure symbolic MLP)
  evaluation/
    metrics.py               # Macro-F1, ROC-AUC, ECE, Brier
scripts/
  run_experiments.py         # all baselines + ablations, multi-seed, in-domain + cross-dataset
  run_indobert.py            # fine-tuned IndoBERT baseline
  analyze_results.py         # LaTeX tables, significance tests, figures
paper/                       # arXiv LaTeX draft (main.tex, sections/, references.bib)
```

## Data

Datasets are not redistributed in this repository. Place them under `data/`:

| Path | Dataset |
|---|---|
| `data/Indonesia Political Hoax Dataset/_combined_all.csv` | Indonesia Political Hoax dataset (columns `cleaned`, `label`, `orig_split`) |
| `data/rahutomo2018/600 news with valid hoax label.csv` | Pratiwi, Asmara & Rahutomo (2017), ICTS, doi:10.1109/ICTS.2017.8265649 |

## Setup

```bash
pip install torch numpy pandas scikit-learn scipy statsmodels pyarrow matplotlib transformers
```

## Running

```bash
python scripts/run_experiments.py --setting original   --seeds 13 21 42 87 100 123 256 512 777 1024
python scripts/run_experiments.py --setting normalized --seeds 13 21 42 87 100 123 256 512 777 1024
python scripts/run_indobert.py --setting original --seed 42
python scripts/analyze_results.py --results results --paper paper
cd paper && latexmk -pdf main.tex
```

## Next steps

- [ ] Run the full experiments (preferably on GPU: the scripts currently train on CPU).
- [ ] Run `analyze_results.py` and write `paper/sections/{abstract,introduction,results,discussion,appendix}.tex` from the real numbers.
- [ ] Add the canonical citation for the Indonesia Political Hoax dataset (TODO in `paper/sections/setup.tex`).
