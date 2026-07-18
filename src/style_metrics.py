"""The guided-revision 10-feature measurement, vendored verbatim.

The impact writer folds in a measure-then-revise loop. This module vendors the
guided-revision tool's exact `compute_features` extractor and its helpers
(`count_syllables`, `_tree_depth`, the constants and PRIMARY_KEYS) rather than
reimplementing them, so the writer's runtime measurement is numerically identical
to that tool's and to the offline eval. spaCy + en_core_web_sm 3.8.0 are pinned in
pyproject to the versions it uses.

Do not edit the algorithm here: it is a copy, and divergence silently decouples the
runtime measurement from the target the summaries are scored against. Re-vendor from
the guided-revision source and bump the pins instead.
"""
import statistics
from functools import lru_cache
from typing import Any, Dict, List, Tuple

VOWELS = set("aeiouy")
NOMINALIZATION_SUFFIXES = ("tion", "ment", "ance", "ence", "ness")

MODAL_DEONTIC = {"shall", "should", "must", "ought", "will"}
PRIMARY_KEYS = [
    "syn_avg_sent_length",
    "syn_sent_length_std",
    "syn_avg_parse_depth",
    "syn_adj_ratio",
    "lex_avg_word_length",
    "lex_avg_syllables_per_word",
    "lex_polysyllabic_ratio",
    "sty_passive_voice_ratio",
    "sty_modal_deontic_per_kw",
    "dis_nominalization_per_kw",
]

def _tree_depth(token) -> int:
    """Number of head-links from `token` up to the sentence root."""
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
    return depth

def count_syllables(word: str) -> int:
    """Heuristic syllable counter: count vowel groups, adjust for silent final 'e'."""
    word = "".join(c for c in word.lower() if c.isalpha())
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for c in word:
        is_vowel = c in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def compute_features(text: str, nlp) -> Dict[str, float]:
    """Compute the 10 primary features on `text`. `nlp` is a loaded spaCy pipeline."""
    doc = nlp(text)
    sentences = list(doc.sents)
    sent_lengths = []
    for sent in sentences:
        sent_word_count = sum(1 for tok in sent if not tok.is_punct and not tok.is_space)
        if sent_word_count > 0:
            sent_lengths.append(sent_word_count)

    if not sent_lengths:
        return {k: 0.0 for k in PRIMARY_KEYS}

    words = [tok for tok in doc if not tok.is_punct and not tok.is_space]
    n_words = len(words)
    if n_words == 0:
        return {k: 0.0 for k in PRIMARY_KEYS}

    avg_sent_length = statistics.mean(sent_lengths)
    sent_length_std = statistics.pstdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

    word_lengths = [len(tok.text) for tok in words]
    avg_word_length = statistics.mean(word_lengths)
    syllables_per_word = [count_syllables(tok.text) for tok in words]
    avg_syllables = statistics.mean(syllables_per_word)
    polysyllabic_count = sum(1 for s in syllables_per_word if s >= 3)
    polysyllabic_ratio = polysyllabic_count / n_words

    max_depths = []
    for sent in sentences:
        depths = [_tree_depth(tok) for tok in sent if not tok.is_punct and not tok.is_space]
        if depths:
            max_depths.append(max(depths))
    avg_parse_depth = statistics.mean(max_depths) if max_depths else 0.0

    adj_ratio = sum(1 for tok in words if tok.pos_ == "ADJ") / n_words

    n_deontic = sum(1 for tok in words if tok.text.lower() in MODAL_DEONTIC)
    modal_deontic_per_kw = (n_deontic / n_words) * 1000

    n_passive_sents = 0
    for sent in sentences:
        has_passive = any(tok.dep_ in ("nsubjpass", "auxpass") for tok in sent)
        if has_passive:
            n_passive_sents += 1
    passive_ratio = n_passive_sents / len(sentences) if sentences else 0.0

    n_nominalizations = sum(
        1 for tok in words
        if tok.pos_ in ("NOUN", "PROPN")
        and len(tok.text) >= 5
        and any(tok.text.lower().endswith(suffix) for suffix in NOMINALIZATION_SUFFIXES)
    )
    nominalizations_per_kw = (n_nominalizations / n_words) * 1000

    return {
        "syn_avg_sent_length": avg_sent_length,
        "syn_sent_length_std": sent_length_std,
        "syn_avg_parse_depth": avg_parse_depth,
        "syn_adj_ratio": adj_ratio,
        "lex_avg_word_length": avg_word_length,
        "lex_avg_syllables_per_word": avg_syllables,
        "lex_polysyllabic_ratio": polysyllabic_ratio,
        "sty_passive_voice_ratio": passive_ratio,
        "sty_modal_deontic_per_kw": modal_deontic_per_kw,
        "dis_nominalization_per_kw": nominalizations_per_kw,
    }

@lru_cache(maxsize=1)
def _load_nlp():
    """Load the pinned spaCy pipeline once (cached). Raises if spaCy/model missing."""
    import spacy
    return spacy.load("en_core_web_sm")

def features(text: str, nlp=None) -> Dict[str, float]:
    """Compute the 10 features on `text`, loading the cached spaCy pipeline if needed."""
    return compute_features(text, nlp or _load_nlp())

def divergences(text: str, target_stats: Dict[str, Any], threshold: float = 0.5, nlp=None
                ) -> List[Tuple[str, float, float, float, float]]:
    """Return (feature, current, target_mean, target_sd, z) rows sorted by |z| desc.

    `target_stats` is the tsbm_impact_statements.json blob (its 'target_stats' map,
    or the bare feature->[mean,sd] map). Only PRIMARY_KEYS present in the target are
    scored. `threshold` is not applied here; the caller filters.
    """
    ts = target_stats.get("target_stats", target_stats) if isinstance(target_stats, dict) else {}
    feats = features(text, nlp)
    rows = []
    for k in PRIMARY_KEYS:
        stat = ts.get(k)
        if not stat or len(stat) < 2:
            continue
        mean, sd = float(stat[0]), float(stat[1])
        z = 0.0 if sd <= 0 else (feats[k] - mean) / sd
        rows.append((k, feats[k], mean, sd, z))
    rows.sort(key=lambda r: -abs(r[4]))
    return rows

def mean_divergences(texts, target_stats: Dict[str, Any], threshold: float = 0.5, nlp=None
                     ) -> List[Tuple[str, float, float, float, float]]:
    """Per-STATEMENT divergence: measure each text as its OWN document and z the MEAN of the
    per-statement feature values against the target. This matches how the TSBM target was built
    (one statement = one document, mean/sd across statements; see data/tsbm_style/build_corpus.py)
    and how build_corpus reports the agent gap, whereas divergences() pools the batch into one
    document and distorts per-statement structure. Same (feature, current, mean, sd, z) rows;
    the caller filters by threshold."""
    ts = target_stats.get("target_stats", target_stats) if isinstance(target_stats, dict) else {}
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    nlp = nlp or _load_nlp()
    feats = [compute_features(t, nlp) for t in texts]
    rows = []
    for k in PRIMARY_KEYS:
        stat = ts.get(k)
        if not stat or len(stat) < 2:
            continue
        mean, sd = float(stat[0]), float(stat[1])
        cur = statistics.mean(f[k] for f in feats)
        z = 0.0 if sd <= 0 else (cur - mean) / sd
        rows.append((k, cur, mean, sd, z))
    rows.sort(key=lambda r: -abs(r[4]))
    return rows
