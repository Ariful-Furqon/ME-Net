import re
import pandas as pd

IPH_PATH = "data/Indonesia Political Hoax Dataset/_combined_all.csv"
RAHUTOMO_PATH = "data/rahutomo2018/600 news with valid hoax label.csv"


def load_iph(dedup: bool = True):
    """Indonesia Political Hoax dataset with its original train/val/test splits.

    With dedup=True, duplicate texts are removed within each split and any validation or
    test text that also occurs in training is dropped, preventing train-test leakage.
    """
    df = pd.read_csv(IPH_PATH).rename(columns={"cleaned": "Text", "label": "Label"})
    df["Text"] = df["Text"].astype(str)
    df = df[df["Text"].str.strip().str.len() > 0]
    splits = {s: df[df["orig_split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if dedup:
        splits["train"] = splits["train"].drop_duplicates("Text").reset_index(drop=True)
        seen = set(splits["train"]["Text"])
        for s in ["val", "test"]:
            d = splits[s].drop_duplicates("Text")
            splits[s] = d[~d["Text"].isin(seen)].reset_index(drop=True)
    return splits


def load_rahutomo():
    """Pratiwi et al. (2017) 600 Indonesian news articles with three-referee voted hoax/valid labels."""
    df = pd.read_csv(RAHUTOMO_PATH, sep=";", encoding="cp1252", encoding_errors="replace")
    df = df.rename(columns={"berita": "Text"})
    df["Text"] = df["Text"].astype(str)
    df["Label"] = (df["tagging"].str.strip().str.lower() == "hoax").astype(int)
    return df[["Text", "Label"]].reset_index(drop=True)


_RE_NONALNUM = re.compile(r"[^a-z0-9\s]+")


def normalize_surface(text: str) -> str:
    """Format-normalised view: lower-cased, punctuation removed, whitespace collapsed.

    A control against source-specific formatting artefacts (casing, punctuation, line breaks).
    """
    return " ".join(_RE_NONALNUM.sub(" ", str(text).lower()).split())
