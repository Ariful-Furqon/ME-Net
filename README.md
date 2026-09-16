# ME-Net: Morpho-Evidential Gated Dual-Stream Network for Indonesian Hoax Detection

ME-Net combines a 1D-CNN over tokens with a 13-dimensional Indonesian linguistic profile
(surface statistics, lexicon-verified affix rates, rumour/attribution evidentials, sensational
markers), fused through a learned complementary gate.

> **Status:** work in progress. This repository holds the code only; the paper draft is kept
> outside it. The 10-seed experiments have been run: in-domain macro-F1 is ~0.976 for ME-Net,
> statistically indistinguishable from a plain CNN and from TF-IDF, and no model transfers to an
> independently labelled corpus.

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
python scripts/run_indobert.py --setting original --seed 42 --epochs 3 --max_len 256 --batch_size 16
python scripts/analyze_results.py --results results --paper <paper directory>
```

Both experiment scripts take `--device` and use CUDA when it is available (`--device cpu` to force CPU).
`analyze_results.py` writes LaTeX tables and figures into the paper directory given by `--paper`.

## License

Code is released under the [MIT License](LICENSE).

`src/data/lexicon_root.txt` is the root-word list from [PySastrawi](https://github.com/har07/PySastrawi) (MIT License, Copyright (c) Hanif Amal Robbani). The datasets are not covered by this license; see their original sources.
