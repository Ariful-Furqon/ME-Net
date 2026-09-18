"""Feasibility gate for v2: does an LLM-extracted claim-level prior add anything?

v1 showed the ME-Net gate collapses to the concatenation regime, so a linear probe
over concatenated features predicts what ME-Net would do without training ME-Net.
If Phi_LLM does not move a probe, it will not move ME-Net either.

Corpus is the Indonesian subset of x-fact: claim-level, no length confound
(61/65/69 chars by label) and no site confound, unlike IPH.
"""

import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.linguistic_profiler import IndonesianLinguisticProfiler  # noqa: E402

csv.field_size_limit(10 ** 9)

URL = "http://localhost:11434/api/generate"
MODEL = "aisingapore/Apertus-SEA-LION-v4-8B-IT:q4_k_m"
WORKERS = 8
OUT = Path("bench_out")

# Claim-level only. Nothing here asks whether the claim is true -- a veracity
# feature would copy the label into Phi and void the experiment.
FEATURES = [
    ("has_specific_quantity", "boolean",
     "Klaim memuat angka, persentase, jumlah uang, atau tanggal yang konkret."),
    ("names_specific_actor", "boolean",
     "Klaim menyebut nama diri orang atau nama lembaga tertentu sebagai pelaku "
     "atau subjek. Sebutan umum seperti \"pemerintah\" atau \"warga\" tidak dihitung."),
    ("is_causal_claim", "boolean",
     "Klaim menyatakan bahwa sesuatu menyebabkan atau mengakibatkan sesuatu yang lain."),
    ("absoluteness", ["none", "some", "strong"],
     "Kemutlakan perumusan klaim: none = tidak ada pemutlakan; some = ada penegasan "
     "sedang; strong = memakai pemutlak seperti \"semua\", \"tidak pernah\", "
     "\"selalu\", \"pasti\"."),
    ("verifiability", ["none", "partial", "specific"],
     "Seberapa mudah klaim dicek pihak lain: none = tidak ada detail yang bisa dicek; "
     "partial = ada sebagian, misal tempat saja atau waktu kabur; specific = ada "
     "detail konkret seperti tanggal, angka, nama dokumen, atau lokasi persis."),
]

ORDINAL_MAPS = {
    "absoluteness": {"none": 0, "some": 1, "strong": 2},
    "verifiability": {"none": 0, "partial": 1, "specific": 2},
}
NAMES = [n for n, _, _ in FEATURES]

SCHEMA = {
    "type": "object",
    "properties": {
        n: ({"type": "boolean"} if t == "boolean" else {"type": "string", "enum": t})
        for n, t, _ in FEATURES
    },
    "required": NAMES,
}

PROMPT = (
    "Tentukan lima properti struktural dari sebuah klaim berbahasa Indonesia.\n\n"
    + "\n".join(f"- {n}: {d}" for n, _, d in FEATURES)
    + "\n\nJawab `false` atau `none` bila tidak ada bukti eksplisit di dalam klaim. "
    "Jangan menebak dan jangan menilai benar-salahnya klaim; nilai hanya bentuknya.\n\n"
    "Contoh 1.\n"
    "Klaim: \"Menteri Kesehatan Budi Sadikin menyatakan stok vaksin naik 30 persen "
    "pada Desember 2024.\"\n"
    "Jawaban: has_specific_quantity=true, names_specific_actor=true, "
    "is_causal_claim=false, absoluteness=\"none\", verifiability=\"specific\"\n\n"
    "Contoh 2.\n"
    "Klaim: \"Semua vaksin menyebabkan kemandulan.\"\n"
    "Jawaban: has_specific_quantity=false, names_specific_actor=false, "
    "is_causal_claim=true, absoluteness=\"strong\", verifiability=\"none\"\n\n"
    "Klaim:\n"
)


class ExtractionFailure(RuntimeError):
    pass


def call(claim):
    body = {
        "model": MODEL, "prompt": PROMPT + claim, "stream": False,
        "format": SCHEMA, "think": False,
        "options": {"num_ctx": 2048, "temperature": 0, "seed": 42},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    # Retry transport errors only: at temperature 0 with a fixed seed, retrying a
    # parse failure reproduces the same unparseable string.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            if attempt == 2:
                raise ExtractionFailure(f"transport: {exc}")
            time.sleep(1.0)
    raw = (payload.get("response") or "").strip() or (payload.get("thinking") or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        raise ExtractionFailure(f"unparseable: {raw[:200]!r}")
    row = {}
    for name, typ, _ in FEATURES:
        v = parsed.get(name)
        if typ == "boolean":
            row[name] = bool(v)
        else:
            m = ORDINAL_MAPS[name]
            if str(v).lower() not in m:
                raise ExtractionFailure(f"{name}={v!r} outside enum")
            row[name] = m[str(v).lower()]
    return row


def load_xfact():
    """Indonesian claims, deduplicated, binary: false vs partly true/misleading."""
    seen, rows = set(), []
    for split in ("train", "dev", "test", "ood", "zeroshot"):
        path = f"data/x-fact/{split}.csv"
        with open(path, encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                if (r.get("language") or "").strip() != "id":
                    continue
                claim = (r.get("claim") or "").strip()
                label = (r.get("label") or "").strip()
                if label not in ("false", "partly true/misleading"):
                    continue
                if len(claim) < 15 or claim in seen:
                    continue
                seen.add(claim)
                rows.append({"claim": claim, "y": int(label == "false"),
                             "site": (r.get("site") or "").strip()})
    return pd.DataFrame(rows)


def extract(df):
    OUT.mkdir(exist_ok=True)
    cache = OUT / "xfact_phi_llm.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(df):
            print(f"[cache] {cache}")
            return cached.reset_index(drop=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(call, df["claim"].tolist()))   # raises on any failure
    el = time.time() - t0
    print(f"[extract] {len(rows)} claims in {el:.0f}s = {el/len(rows):.2f}s/claim")
    out = pd.DataFrame(rows, columns=NAMES)
    out.to_parquet(cache)
    return out


def probe(X, y, seeds=(0, 1, 2, 3, 4), fractions=(0.05, 0.1, 0.25, 0.5, 1.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for frac in fractions:
        scores = []
        for seed in seeds:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y)
            if frac < 1.0:
                idx, _ = next(StratifiedShuffleSplit(
                    n_splits=1, train_size=frac, random_state=seed).split(Xtr, ytr))
                Xtr, ytr = Xtr[idx], ytr[idx]
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(Xtr, ytr)
            scores.append(f1_score(yte, clf.predict(Xte), average="macro"))
        out[frac] = (float(np.mean(scores)), float(np.std(scores)))
    return out


def main():
    df = load_xfact()
    print(f"x-fact ID binary: n={len(df)}  false={int(df.y.sum())}  "
          f"partly/misleading={int((1-df.y).sum())}")
    print(f"claim length by label: false median={int(df[df.y==1].claim.str.len().median())}  "
          f"other median={int(df[df.y==0].claim.str.len().median())}")

    phi_llm = extract(df)

    prof = IndonesianLinguisticProfiler()
    phi_v1 = pd.DataFrame([prof.extract_features_single(c) for c in df["claim"]])
    print(f"Phi_v1 dims={phi_v1.shape[1]}  Phi_LLM dims={phi_llm.shape[1]}")

    y = df["y"].values
    sets = {
        "length only (sanity)": df[["claim"]].assign(n=df.claim.str.len())[["n"]].values.astype(float),
        "Phi_v1 (13)": phi_v1.values.astype(float),
        "Phi_LLM (5)": phi_llm.values.astype(float),
        "Phi_v1 + Phi_LLM (18)": np.hstack([phi_v1.values, phi_llm.values]).astype(float),
    }

    print("\n=== linear probe, macro-F1 (mean +/- sd over 5 splits) ===")
    header = "  ".join(f"{f:>12}" for f in (0.05, 0.1, 0.25, 0.5, 1.0))
    print(f"{'feature set':<24}{header}")
    results = {}
    for name, X in sets.items():
        r = probe(X, y)
        results[name] = r
        cells = "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in r.values())
        print(f"{name:<24}{cells}")

    print("\n=== redundancy: max |corr| of each Phi_LLM feature vs any Phi_v1 feature ===")
    for c in NAMES:
        cors = [abs(np.corrcoef(phi_llm[c].astype(float), phi_v1[v].astype(float))[0, 1])
                for v in phi_v1.columns]
        cors = [x for x in cors if not np.isnan(x)]
        best = int(np.argmax(cors))
        print(f"  {c:<24} max|r|={max(cors):.3f}  (vs {phi_v1.columns[best]})")

    print("\n=== degeneracy check on Phi_LLM ===")
    for c in NAMES:
        vc = phi_llm[c].value_counts(normalize=True).to_dict()
        print(f"  {c:<24} {({k: round(v,3) for k,v in vc.items()})}")

    (OUT / "probe_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
