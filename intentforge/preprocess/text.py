"""Lightweight text preprocessing shared by sklearn-family backends.

We deliberately keep preprocessing minimal (lowercase + collapse whitespace):
TF-IDF + the downstream model already capture the signal, and a heavy cleaner
would hide the backends' relative strengths. The module is a single place to
tune if a project needs stemming/lemmatization later.
"""

from __future__ import annotations

import re
from typing import List

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Normalize a single text string."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _WS.sub(" ", text).strip()
    return text


def clean_all(texts: List[str]) -> List[str]:
    return [clean(t) for t in texts]


def build_vectorizer(cfg):
    """Build a configured ``TfidfVectorizer`` (shared by linear + xgboost)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_tuple(),
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents="unicode",
    )
