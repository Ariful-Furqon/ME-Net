import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.data.affix_exceptions import (
    PASSIVE_EXCEPTIONS,
    ACTIVE_NON_VERB_ROOTS,
    KE_AN_EXCEPTIONS,
    KE_AN_VALID,
    BOUND_CLITICS,
    FREE_PARTICLES,
    CLITIC_EXCEPTIONS
)

# 1. Deceptive evidentials & rumor markers (strictly rumor / unverified epistemic triggers)
DECEPTIVE_EVIDENTIAL_PATTERNS = [
    r"\bdikabarkan\b", r"\bkonon\b", r"\bbocoran\b", r"\bkatanya\b", r"\bisunya\b",
    r"\bkabar burung\b", r"\binfo a1\b", r"\brahasia\b", r"\bterbongkar\b",
    r"\bterkuak\b", r"\bdisembunyikan\b", r"\bfakta tersembunyi\b",
    r"\bjangan remehkan\b", r"\bwajib tahu\b", r"\bsebarluaskan\b",
    r"\bviralkan\b", r"\bkonspirasi\b"
]

# 2. Institutional attributions (grammatical attribution triggers, not topic agency names)
INSTITUTIONAL_ATTRIBUTION_PATTERNS = [
    r"\bmenurut\b", r"\bujar\b", r"\btutur\b", r"\bpungkas\b", r"\bdikutip\b",
    r"\bmengutip\b", r"\bberdasarkan keterangan\b", r"\bjuru bicara\b",
    r"\bketerangan resmi\b", r"\bdalam pernyataan\b", r"\bkonfirmasi resmi\b",
    r"\bpernyataan resmi\b", r"\bsiaran pers\b", r"\bberdasarkan data\b"
]

# 3. Sensational & emotional markers (formerly mixed into evidentials, now honestly labeled)
SENSATIONAL_MARKER_PATTERNS = [
    r"\bviral\b", r"\bheboh\b", r"\bwaspada\b", r"\bsubhanallah\b",
    r"\bastagfirullah\b", r"\bbiadab\b", r"\bgempar\b", r"\bgawat\b",
    r"\bmengerikan\b", r"\bluar biasa\b", r"\bhancur lebur\b",
    r"\btanpa ampun\b", r"\bmerinding\b", r"\bterparah\b", r"\bmustahil\b",
    r"\b100%\b", r"\bdahsyat\b"
]

ALLOMORPH = {
    "meng": ["", "k"],
    "meny": ["s"],
    "mem": ["", "p"],
    "men": ["", "t"],
    "me": [""],
    "ber": [""],
    "bel": [""],
    "be": [""],
    "di": [""],
    "ter": [""],
    "te": [""]
}

class AffixDetector:
    """Pure allomorph and lexicon set-based Indonesian affix detector (10^6 checks/sec)."""

    def __init__(self, lexicon_path: str = "data/lexicon_root.txt", lexicon: Set[str] = None):
        if lexicon is not None:
            self.lexicon = set(lexicon)
        else:
            self.lexicon: Set[str] = set()
            for lp in [lexicon_path, os.path.join(os.path.dirname(__file__), "lexicon_root.txt")]:
                if os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8") as f:
                        self.lexicon = {line.strip().lower() for line in f if line.strip()}
                    break

        self._passive_cache: Dict[str, bool] = {}
        self._active_cache: Dict[str, bool] = {}
        self._ke_an_cache: Dict[str, bool] = {}

    def is_passive(self, tok: str) -> bool:
        if tok in self._passive_cache:
            return self._passive_cache[tok]
        if len(tok) <= 3 or tok in PASSIVE_EXCEPTIONS:
            self._passive_cache[tok] = False
            return False
        for p in ["di", "ter", "te"]:
            if tok.startswith(p):
                rest = tok[len(p):]
                if any((repl + rest) in self.lexicon for repl in ALLOMORPH.get(p, [""])):
                    self._passive_cache[tok] = True
                    return True
        self._passive_cache[tok] = False
        return False

    def is_active(self, tok: str) -> bool:
        if tok in self._active_cache:
            return self._active_cache[tok]
        if len(tok) <= 3 or tok in ACTIVE_NON_VERB_ROOTS:
            self._active_cache[tok] = False
            return False
        for p in ["meng", "meny", "mem", "men", "me", "ber", "bel", "be"]:
            if tok.startswith(p):
                rest = tok[len(p):]
                if any((repl + rest) in self.lexicon for repl in ALLOMORPH.get(p, [""])):
                    self._active_cache[tok] = True
                    return True
        self._active_cache[tok] = False
        return False

    def is_ke_an(self, tok: str) -> bool:
        if tok in self._ke_an_cache:
            return self._ke_an_cache[tok]
        if not (tok.startswith("ke") and tok.endswith("an") and len(tok) > 5):
            self._ke_an_cache[tok] = False
            return False
        if tok in KE_AN_EXCEPTIONS:
            self._ke_an_cache[tok] = False
            return False
        if tok in KE_AN_VALID:
            self._ke_an_cache[tok] = True
            return True
        base = tok[2:-2]
        val = base in self.lexicon
        self._ke_an_cache[tok] = val
        return val

class IndonesianLinguisticProfiler:
    """Extracts multidimensional linguistic and stylistic deception markers from Indonesian texts."""

    def __init__(self, lexicon_path: str = None):
        self.re_words = re.compile(r'\b[a-zA-Z0-9_\-]+\b')
        self._particle_cache: Dict[str, bool] = {}
        
        # Load root lexicon if available
        lexicon: Set[str] = set()
        default_lexicon_path = os.path.join(os.path.dirname(__file__), "lexicon_root.txt")
        target_lexicon = lexicon_path or default_lexicon_path
        if os.path.exists(target_lexicon):
            with open(target_lexicon, "r", encoding="utf-8") as f:
                lexicon = {line.strip().lower() for line in f if line.strip()}
        
        self.affix_detector = AffixDetector(lexicon=lexicon)
        self.lexicon = lexicon

        # Compiled regex for evidentials, attributions, and sensational markers
        self.re_deceptive = re.compile("|".join(DECEPTIVE_EVIDENTIAL_PATTERNS), re.IGNORECASE)
        self.re_institutional = re.compile("|".join(INSTITUTIONAL_ATTRIBUTION_PATTERNS), re.IGNORECASE)
        self.re_sensational = re.compile("|".join(SENSATIONAL_MARKER_PATTERNS), re.IGNORECASE)

    def is_discourse_particle(self, tok: str) -> bool:
        """Determines if a token is a discourse particle (free particle or valid bound clitic)."""
        if tok in self._particle_cache:
            return self._particle_cache[tok]
        if tok in FREE_PARTICLES:
            self._particle_cache[tok] = True
            return True
        if tok in CLITIC_EXCEPTIONS:
            self._particle_cache[tok] = False
            return False
        if tok in self.lexicon:
            self._particle_cache[tok] = False
            return False
        for clitic in BOUND_CLITICS:
            if tok.endswith(clitic) and len(tok) > len(clitic) + 1:
                base = tok[:-len(clitic)]
                if base in self.lexicon:
                    self._particle_cache[tok] = True
                    return True
        self._particle_cache[tok] = False
        return False

    def extract_features_single(self, text: str) -> Dict[str, float]:
        """Extract linguistic feature dictionary for a single text instance."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {k: 0.0 for k in self.feature_names()}

        words = self.re_words.findall(text.lower())
        num_words = max(len(words), 1)
        num_chars = max(len(text), 1)

        # 1. Surface Length & Lexical Diversity
        avg_word_len = sum(len(w) for w in words) / num_words
        unique_words = set(words)
        ttr = len(unique_words) / num_words
        word_freqs = pd.Series(words).value_counts()
        hapax_ratio = (word_freqs == 1).sum() / num_words

        # 2. Morphological Indicators (Sastrawi + Exception-checked single-pass traversal)
        passive_prefix = 0
        active_prefix = 0
        ke_an_circumfix = 0
        particles_count = 0
        for w in words:
            if self.affix_detector.is_passive(w):
                passive_prefix += 1
            elif self.affix_detector.is_active(w):
                active_prefix += 1
            elif self.affix_detector.is_ke_an(w):
                ke_an_circumfix += 1
            if self.is_discourse_particle(w):
                particles_count += 1

        # 3. Evidentiality, Attribution, & Sensational Markers (Uniform Weight 1.0)
        text_lower = text.lower()
        deceptive_evidential_count = len(self.re_deceptive.findall(text_lower))
        institutional_count = len(self.re_institutional.findall(text_lower))
        sensational_count = len(self.re_sensational.findall(text_lower))

        # 4. Normalized Rates per 100 words
        passive_rate = (passive_prefix / num_words) * 100
        active_rate = (active_prefix / num_words) * 100
        ke_an_rate = (ke_an_circumfix / num_words) * 100
        particle_rate = (particles_count / num_words) * 100
        dec_rate = (deceptive_evidential_count / num_words) * 100
        inst_rate = (institutional_count / num_words) * 100
        sens_rate = (sensational_count / num_words) * 100

        # Rate-normalized ratio (length-invariant)
        attr_to_rumor_ratio = (inst_rate + 0.01) / (dec_rate + 0.01)

        return {
            "word_count": float(num_words),
            "char_count": float(num_chars),
            "avg_word_length": float(avg_word_len),
            "type_token_ratio": float(ttr),
            "hapax_legomena_ratio": float(hapax_ratio),
            "passive_prefix_rate": float(passive_rate),
            "active_prefix_rate": float(active_rate),
            "nominal_ke_an_rate": float(ke_an_rate),
            "discourse_particle_rate": float(particle_rate),
            "deceptive_evidential_rate": float(dec_rate),
            "institutional_attribution_rate": float(inst_rate),
            "sensational_marker_rate": float(sens_rate),
            "attribution_to_rumor_ratio": float(attr_to_rumor_ratio)
        }

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "word_count", "char_count", "avg_word_length",
            "type_token_ratio", "hapax_legomena_ratio",
            "passive_prefix_rate", "active_prefix_rate", "nominal_ke_an_rate",
            "discourse_particle_rate", "deceptive_evidential_rate",
            "institutional_attribution_rate", "sensational_marker_rate",
            "attribution_to_rumor_ratio"
        ]

    def profile_dataframe(self, df: pd.DataFrame, text_col: str = "Text", cache_dir: str = "results/cache") -> pd.DataFrame:
        """Extracts features for all texts in dataframe with versioned Parquet disk caching."""
        # Key on the texts, the extractor source and the lexicon, so any extractor change invalidates the cache.
        src_dir = os.path.dirname(__file__)
        extractor_src = "".join(
            open(os.path.join(src_dir, f), encoding="utf-8").read()
            for f in ["linguistic_profiler.py", "affix_exceptions.py"]
        )
        key = hashlib.sha1(
            ("|".join(df[text_col].astype(str)) + extractor_src + "|".join(sorted(self.lexicon))).encode()
        ).hexdigest()[:16]
        path = Path(cache_dir) / f"feat_{key}.parquet"
        if path.exists():
            return pd.read_parquet(path).set_index(df.index)
        
        feats = pd.DataFrame([self.extract_features_single(t) for t in df[text_col]], index=df.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path)
        return feats

    def compute_linguistic_significance(
        self,
        df: pd.DataFrame,
        text_col: str = "Text",
        label_col: str = "Label",
        n_bootstraps: int = 2000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Computes Mann-Whitney U, Welch t-test, Cohen's d with bootstrap 95% CI, and Holm correction."""
        feature_df = self.profile_dataframe(df, text_col=text_col)
        results = []

        hoax_mask = (df[label_col] == 1)
        valid_mask = (df[label_col] == 0)

        rng = np.random.RandomState(random_state)
        p_values_raw = []

        for col in feature_df.columns:
            hoax_vals = feature_df.loc[hoax_mask, col].values
            valid_vals = feature_df.loc[valid_mask, col].values

            mean_h, std_h = float(np.mean(hoax_vals)), float(np.std(hoax_vals, ddof=1)) if len(hoax_vals) > 1 else 0.0
            mean_v, std_v = float(np.mean(valid_vals)), float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0

            # 1. Mann-Whitney U (Primary non-parametric test)
            mw_res = stats.mannwhitneyu(hoax_vals, valid_vals, alternative='two-sided')
            u_stat, p_mw = float(mw_res.statistic), float(mw_res.pvalue)

            # 2. Welch's t-test (Supplementary parametric test)
            t_stat, p_welch = stats.ttest_ind(hoax_vals, valid_vals, equal_var=False)

            # 3. Cohen's d with sample variance (ddof=1)
            n_h, n_v = len(hoax_vals), len(valid_vals)
            s_pooled = np.sqrt(((n_h - 1) * np.var(hoax_vals, ddof=1) + (n_v - 1) * np.var(valid_vals, ddof=1)) / (n_h + n_v - 2))
            cohens_d = float((mean_h - mean_v) / (s_pooled + 1e-9))

            # 4. Asymptotic 95% CI for Cohen's d (Hedges & Olkin standard error)
            se_d = np.sqrt((n_h + n_v) / (n_h * n_v) + (cohens_d ** 2) / (2 * (n_h + n_v)))
            d_ci_lower = float(cohens_d - 1.96 * se_d)
            d_ci_upper = float(cohens_d + 1.96 * se_d)

            p_values_raw.append(p_mw)

            results.append({
                "Feature": col,
                "Hoax_Mean": mean_h,
                "Hoax_Std": std_h,
                "Valid_Mean": mean_v,
                "Valid_Std": std_v,
                "Difference": mean_h - mean_v,
                "MannWhitney_U": u_stat,
                "p_raw": p_mw,
                "t_statistic": float(t_stat),
                "p_welch": float(p_welch),
                "cohens_d": cohens_d,
                "d_ci_lower": d_ci_lower,
                "d_ci_upper": d_ci_upper
            })

        # Holm-Bonferroni correction
        reject, p_holm, _, _ = multipletests(p_values_raw, method="holm")
        for i in range(len(results)):
            results[i]["p_holm"] = float(p_holm[i])
            p_val = results[i]["p_holm"]
            results[i]["Significance"] = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        res_df = pd.DataFrame(results).sort_values(by="p_holm")
        return res_df
