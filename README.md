# ME-Net: Morpho-Evidential Gated Dual-Stream Network for Indonesian Hoax Detection

ME-Net combines a 1D-CNN over tokens with a 13-dimensional Indonesian linguistic profile
(surface statistics, lexicon-verified affix rates, rumour/attribution evidentials, sensational
markers), fused through a learned complementary gate whose average is an interpretable
text-reliance score.

> **Status:** research code, work in progress. This repository holds the code only; the paper draft
> lives outside it. The experiments below have been run and their headline results are summarised
> in [Findings](#findings) — including a negative one about the gate.

## Repository layout

```
src/
  data/
    linguistic_profiler.py  # 13-feature document-level profile + significance tests
    claim_profiler.py       # 15-feature claim-level profile (Indonesian and English)
    affix_exceptions.py     # exception lists for the affix/clitic detector
    lexicon_root.txt        # 29,931 Indonesian root words (Sastrawi kata-dasar list)
    tokenizer.py            # word-level tokenizer
    loaders.py              # IPH, Rahutomo-600, X-FACT, LIAR2, format normalisation
  models/
    me_fusion.py            # ME-Net and the gated cross-modal fusion block
    ablation_study.py       # ablation variants (naive concat, pure CNN, pure symbolic MLP)
    fusion_variants.py      # feature conditioners and alternative gate designs
  evaluation/metrics.py     # macro-F1, ROC-AUC, ECE, Brier
scripts/
  run_experiments.py        # baselines + ablations on IPH, multi-seed, in-domain + cross-dataset
  run_indobert.py           # fine-tuned IndoBERT baseline
  run_claims.py             # learning curves on short-claim corpora (X-FACT id, LIAR2)
  diagnostics.py            # learning curve, length-matched control, cross-corpus CV, domain probe
  run_variants.py           # gate/feature variant study, scored on validation only
  analyze_results.py        # LaTeX tables, significance tests, figures
```

## Data

Datasets are not redistributed here. Place them under `data/`:

| Path | Dataset |
|---|---|
| `data/Indonesia Political Hoax Dataset/_combined_all.csv` | Indonesia Political Hoax (columns `cleaned`, `label`, `orig_split`) |
| `data/rahutomo2018/600 news with valid hoax label.csv` | Pratiwi, Asmara & Rahutomo (2017), ICTS, doi:10.1109/ICTS.2017.8265649 |
| `data/x-fact/{train,dev,test,ood,zeroshot}.csv` | [X-FACT](https://huggingface.co/datasets/utahnlp/x-fact) (MIT), 25 languages; the Indonesian subset is used |
| `data/liar/{train,validation,test}.csv` | [LIAR2](https://huggingface.co/datasets/chengxuphd/liar2) (Apache-2.0), English PolitiFact claims |

The loaders use claim text only. X-FACT `evidence_*` and LIAR2 `justification` are written by the
fact-checker together with the verdict, so they leak the label and are dropped.

`load_iph` de-duplicates: it removes repeated texts within each split and any validation/test document
that also occurs in training (153 documents), which the published split does not do.

## Setup

```bash
pip install torch numpy pandas scikit-learn scipy statsmodels pyarrow matplotlib transformers datasets
```

For GPU, install a CUDA build of torch (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu130`).
Every training script takes `--device` and uses CUDA when available; pass `--device cpu` to force CPU.

## Running

```bash
# main comparison on IPH: baselines, ablations, 10 seeds, raw and format-normalised text
python scripts/run_experiments.py --setting original   --seeds 13 21 42 87 100 123 256 512 777 1024
python scripts/run_experiments.py --setting normalized --seeds 13 21 42 87 100 123 256 512 777 1024

# transformer baseline
python scripts/run_indobert.py --setting original --seed 42 --epochs 3 --max_len 256 --batch_size 16

# why the in-domain numbers look the way they do
python scripts/diagnostics.py --seeds 13 21 42 87 100

# short-claim corpora, learning curves in two languages
python scripts/run_claims.py --dataset both --seeds 13 21 42 87 100

# gate and feature-conditioning variants (validation only)
python scripts/run_variants.py --dataset iph --train_size 719 --seeds 13 21 42 87 100

# tables and figures for the paper directory
python scripts/analyze_results.py --results results --paper <paper directory>
```

Results are written as JSONL under `results/`, one row per model, seed and evaluation split.

## Findings

Measured, not claimed. All figures are means over 5–10 seeds.

- **On IPH the added structure does not improve accuracy.** ME-Net reaches macro-F1 0.976±0.005 against
  0.974±0.003 for a text-only CNN and 0.975 for TF-IDF; no ablation difference survives Holm correction.
- **The gate collapses.** Its per-document average stays within [0.26, 0.45] with a spread of 0.01–0.02,
  i.e. it learns a nearly constant mixing weight — the degenerate case of the expressiveness result — and
  the gated model is statistically indistinguishable from plain concatenation, and worse than it on LIAR2
  (−0.66 F1, 0/10 seeds, p=0.002). Nine repair variants were tried on validation data; none helped.
- **The benchmark is the limiting factor.** Two length features alone reach 0.638 on IPH; after matching
  the classes on length TF-IDF still reaches 0.978, so the shortcut is source vocabulary. Every model,
  including IndoBERT, drops to 0.52–0.56 on an independently labelled corpus with ECE rising tenfold.
- **Where the prior does pay off.** With 5% of the IPH training data ME-Net beats the CNN by 8.2 F1
  points (5/5 seeds) and by 4.3 points at 10%; the gap closes from 25% upwards. On the out-of-site split
  of X-FACT Indonesian, symbolic features alone are the most robust model (0.519 vs 0.451 for TF-IDF).
- **Which linguistic hypothesis held.** Institutional attribution is markedly rarer in hoax text
  (d = −0.62), while rumour evidentials do not separate the classes at all (d = −0.009).

## License

Code is released under the [MIT License](LICENSE).

`src/data/lexicon_root.txt` is the root-word list from [PySastrawi](https://github.com/har07/PySastrawi)
(MIT License, Copyright (c) Hanif Amal Robbani). The datasets are not covered by this license; see their
original sources.
