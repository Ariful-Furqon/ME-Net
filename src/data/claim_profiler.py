import re
from typing import Dict, List

import numpy as np
import pandas as pd

# --- Indonesian lexicons -------------------------------------------------------------------
ID_ATTRIBUTION = [r"\bmenurut\b", r"\bujar\b", r"\btutur\b", r"\bkata\b", r"\bmengatakan\b",
                  r"\bmenyatakan\b", r"\bpernyataan\b", r"\bjuru bicara\b", r"\bdikutip\b",
                  r"\bmengutip\b", r"\bsiaran pers\b", r"\bkonfirmasi\b"]
ID_HEARSAY = [r"\bkonon\b", r"\bkabarnya\b", r"\bdikabarkan\b", r"\bkatanya\b", r"\bdiduga\b",
              r"\bdisebut-sebut\b", r"\bisu\b", r"\bberedar\b", r"\bviral\b", r"\bbocoran\b",
              r"\bdiklaim\b", r"\bklaim\b", r"\bhoaks?\b", r"\bkabar burung\b"]
ID_HEDGE = [r"\bmungkin\b", r"\bdiduga\b", r"\bsepertinya\b", r"\bdisinyalir\b", r"\bkemungkinan\b",
            r"\bdikabarkan\b", r"\bdiperkirakan\b", r"\bbisa jadi\b"]
ID_NEGATION = [r"\btidak\b", r"\bbukan\b", r"\btak\b", r"\btanpa\b", r"\bbelum\b", r"\bjangan\b",
               r"\bgagal\b", r"\bdilarang\b"]
ID_SENSATIONAL = [r"\bheboh\b", r"\bmengerikan\b", r"\bgempar\b", r"\bwaspada\b", r"\bdarurat\b",
                  r"\bbahaya\b", r"\bmengejutkan\b", r"\bterbongkar\b", r"\bterkuak\b",
                  r"\bskandal\b", r"\bmematikan\b", r"\bluar biasa\b"]
ID_ABSOLUTE = [r"\bsemua\b", r"\bseluruh\b", r"\bselalu\b", r"\btidak pernah\b", r"\bhanya\b",
               r"\bsatu-satunya\b", r"\bpaling\b", r"\bter\w+kan\b", r"\bsetiap\b", r"\bseluruhnya\b"]

# --- English lexicons ----------------------------------------------------------------------
EN_ATTRIBUTION = [r"\baccording to\b", r"\bsaid\b", r"\bsays\b", r"\bstated\b", r"\btold\b",
                  r"\bannounced\b", r"\btestified\b", r"\bspokes(?:man|woman|person)\b",
                  r"\bin a statement\b", r"\bquoted\b", r"\bwrote\b", r"\btestimony\b"]
EN_HEARSAY = [r"\breportedly\b", r"\ballegedly\b", r"\brumou?rs?\b", r"\bsources say\b",
              r"\bpurportedly\b", r"\bsupposedly\b", r"\bapparently\b", r"\bclaims?\b",
              r"\balleges?\b", r"\bviral\b", r"\bleaked\b"]
EN_HEDGE = [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bwould\b", r"\bpossibly\b", r"\blikely\b",
            r"\bsuggests?\b", r"\bappears?\b", r"\bseems?\b", r"\bpotentially\b", r"\bnearly\b",
            r"\balmost\b"]
EN_NEGATION = [r"\bnot\b", r"n't\b", r"\bno\b", r"\bnever\b", r"\bnone\b", r"\bnobody\b",
               r"\bwithout\b", r"\bfailed\b", r"\bdenied\b"]
EN_SENSATIONAL = [r"\bshocking\b", r"\boutrageous\b", r"\bdisaster\b", r"\bdestroy(?:ed|ing)?\b",
                  r"\bmassive\b", r"\bunprecedented\b", r"\bhistoric\b", r"\bhuge\b", r"\bcrisis\b",
                  r"\bdeadly\b", r"\bscandal\b", r"\bexplosive\b"]
EN_ABSOLUTE = [r"\ball\b", r"\bevery\b", r"\balways\b", r"\bnever\b", r"\bonly\b", r"\bentire\b",
               r"\bzero\b", r"\bworst\b", r"\bbest\b", r"\bfirst\b", r"\bmost\b", r"\bhighest\b",
               r"\blowest\b"]

EN_BE = r"(?:is|are|was|were|be|been|being|has been|have been|had been)"
EN_PASSIVE = re.compile(EN_BE + r"\s+(?:\w+ly\s+)?(\w+(?:ed|en|wn|ught| built|paid|made|told|held|cut|set|put|sent))\b", re.I)
EN_NOMINAL = re.compile(r"\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence)\b", re.I)

# --- Shared surface patterns ---------------------------------------------------------------
RE_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
RE_DIGIT = re.compile(r"\d")
RE_MAGNITUDE = re.compile(r"%|\bpercent\b|\bpersen\b|\bmillion\b|\bbillion\b|\btrillion\b|\bjuta\b|\bmiliar\b|\btriliun\b|\brp\b|\$", re.I)
RE_QUOTE = re.compile(r"[\"“”‘’']")
RE_EXCLAIM = re.compile(r"[!?]")

FEATURE_NAMES = [
    "word_count", "avg_word_length", "uppercase_word_ratio",
    "digit_count", "magnitude_count", "quote_count", "exclaim_count",
    "attribution_count", "hearsay_count", "hedge_count",
    "negation_count", "sensational_count", "absolute_count",
    "passive_count", "nominalization_count",
]


def _compile(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


class ClaimProfiler:
    """Language-parameterised claim-level profiler producing the 15 features in FEATURE_NAMES."""

    def __init__(self, language: str = "id", affix_detector=None):
        if language not in ("id", "en"):
            raise ValueError("language must be 'id' or 'en'")
        self.language = language
        if language == "id":
            groups = (ID_ATTRIBUTION, ID_HEARSAY, ID_HEDGE, ID_NEGATION, ID_SENSATIONAL, ID_ABSOLUTE)
            if affix_detector is None:
                from src.data.affix_exceptions import (ACTIVE_NON_VERB_ROOTS, KE_AN_EXCEPTIONS,  # noqa: F401
                                                       KE_AN_VALID, PASSIVE_EXCEPTIONS)
                from src.data.linguistic_profiler import AffixDetector
                affix_detector = AffixDetector()
            self.affix = affix_detector
        else:
            groups = (EN_ATTRIBUTION, EN_HEARSAY, EN_HEDGE, EN_NEGATION, EN_SENSATIONAL, EN_ABSOLUTE)
            self.affix = None
        (self.re_attr, self.re_hearsay, self.re_hedge,
         self.re_neg, self.re_sens, self.re_abs) = (_compile(g) for g in groups)

    # -- morphology ---------------------------------------------------------------------
    def _passive_and_nominal(self, text: str, words: List[str]) -> tuple:
        if self.language == "id":
            passive = sum(1 for w in words if self.affix.is_passive(w.lower()))
            nominal = sum(1 for w in words if self.affix.is_ke_an(w.lower()))
        else:
            passive = len(EN_PASSIVE.findall(text))
            nominal = len(EN_NOMINAL.findall(text))
        return float(passive), float(nominal)

    def extract_features_single(self, text: str) -> Dict[str, float]:
        text = "" if not isinstance(text, str) else text
        words = RE_WORDS.findall(text)
        n = max(len(words), 1)
        passive, nominal = self._passive_and_nominal(text, words)
        return {
            "word_count": float(len(words)),
            "avg_word_length": float(sum(len(w) for w in words) / n),
            "uppercase_word_ratio": float(sum(1 for w in words if w[:1].isupper()) / n),
            "digit_count": float(len(RE_DIGIT.findall(text))),
            "magnitude_count": float(len(RE_MAGNITUDE.findall(text))),
            "quote_count": float(len(RE_QUOTE.findall(text))),
            "exclaim_count": float(len(RE_EXCLAIM.findall(text))),
            "attribution_count": float(len(self.re_attr.findall(text))),
            "hearsay_count": float(len(self.re_hearsay.findall(text))),
            "hedge_count": float(len(self.re_hedge.findall(text))),
            "negation_count": float(len(self.re_neg.findall(text))),
            "sensational_count": float(len(self.re_sens.findall(text))),
            "absolute_count": float(len(self.re_abs.findall(text))),
            "passive_count": passive,
            "nominalization_count": nominal,
        }

    @staticmethod
    def feature_names() -> List[str]:
        return list(FEATURE_NAMES)

    def profile(self, texts) -> pd.DataFrame:
        return pd.DataFrame([self.extract_features_single(t) for t in texts], columns=FEATURE_NAMES)

    def zero_rates(self, texts) -> pd.Series:
        """Share of documents for which each feature is exactly zero (a degeneracy check)."""
        f = self.profile(texts)
        return (f == 0).mean()


def significance(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mann-Whitney U with Holm correction and Cohen's d, as in the document-level profiler."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    rows, praw = [], []
    for c in features.columns:
        a = features.loc[labels == 1, c].values
        b = features.loc[labels == 0, c].values
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)) + 1e-9
        d = float((a.mean() - b.mean()) / sp)
        se = np.sqrt((na + nb) / (na * nb) + d ** 2 / (2 * (na + nb)))
        praw.append(float(u.pvalue))
        rows.append({"Feature": c, "Label0_Mean": float(b.mean()), "Label1_Mean": float(a.mean()),
                     "cohens_d": d, "d_ci_lower": d - 1.96 * se, "d_ci_upper": d + 1.96 * se,
                     "p_raw": float(u.pvalue), "zero_rate": float((features[c] == 0).mean())})
    for r, p in zip(rows, multipletests(praw, method="holm")[1]):
        r["p_holm"] = float(p)
    return pd.DataFrame(rows).sort_values("p_holm")
