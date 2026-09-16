import re
from collections import Counter
from typing import List
import numpy as np


class SimpleTokenizer:

    def __init__(self, max_vocab: int = 30000, max_len: int = 256, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.min_freq = min_freq
        self.re_tok = re.compile(r"[a-z0-9]+")
        self.word2idx = {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        return self.re_tok.findall(str(text).lower())

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        counts = Counter(tok for t in texts for tok in self.tokenize(t))
        for tok, c in counts.most_common(self.max_vocab - 2):
            if c < self.min_freq:
                break
            self.word2idx[tok] = len(self.word2idx)
        return self

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.word2idx.get(tok, 1) for tok in self.tokenize(t)][: self.max_len]
            out[i, : len(ids)] = ids
        return out
