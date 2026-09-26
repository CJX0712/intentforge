"""Data loaders implementing :class:`IDataLoader`.

The default loader is fully synthetic (no network). CSV and 20-Newsgroups
loaders are provided as drop-in alternatives and degrade gracefully when the
optional dependency is missing.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

from intentforge.core.config import Config
from intentforge.core.errors import DataError
from intentforge.data.generator import generate_texts, make_splits


class BuiltinLoader:
    """Deterministic synthetic intent dataset (default; offline)."""

    name = "builtin"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> Dict[str, Any]:
        texts, labels = generate_texts(self.cfg)
        splits = make_splits(texts, labels, self.cfg)
        return splits


class CsvLoader:
    """Load a CSV file with ``text`` and ``label`` columns."""

    name = "csv"

    def __init__(self, cfg: Config, path: str, text_col: str = "text", label_col: str = "label"):
        self.cfg = cfg
        self.path = path
        self.text_col = text_col
        self.label_col = label_col

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            raise DataError(f"csv not found: {self.path}", code="E200")
        texts: List[str] = []
        labels: List[str] = []
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                t = row.get(self.text_col)
                l = row.get(self.label_col)
                if t and l:
                    texts.append(t)
                    labels.append(l)
        if not texts:
            raise DataError("csv produced 0 rows", code="E200")
        self.cfg.label_names = sorted(set(labels))[: self.cfg.n_classes]
        return make_splits(texts, labels, self.cfg)


class NewsgroupsLoader:
    """Optional 20-Newsgroups loader (requires ``sklearn`` datasets)."""

    name = "newsgroups"

    def __init__(self, cfg: Config, categories=None, subset_size: int = 2000):
        self.cfg = cfg
        self.categories = categories
        self.subset_size = subset_size

    def load(self) -> Dict[str, Any]:
        try:
            from sklearn.datasets import fetch_20newsgroups
        except Exception as exc:  # pragma: no cover - sklearn always present
            raise DataError("sklearn unavailable", code="E200", cause=exc) from exc
        data = fetch_20newsgroups(
            subset="all", categories=self.categories, remove=("headers", "footers", "quotes")
        )
        texts = [str(t) for t in data["data"]]
        labels = [str(l) for l in data["target_names"]]
        y = [labels[i] for i in data["target"]]
        if self.subset_size and len(texts) > self.subset_size:
            rng = __import__("random").Random(self.cfg.random_seed)
            idx = rng.sample(range(len(texts)), self.subset_size)
            texts = [texts[i] for i in idx]
            y = [y[i] for i in idx]
        self.cfg.label_names = sorted(set(y))[: self.cfg.n_classes]
        return make_splits(texts, y, self.cfg)


def get_loader(cfg: Config, source: str = "builtin", **kwargs) -> Any:
    """Factory returning the configured loader."""
    if source == "builtin":
        return BuiltinLoader(cfg)
    if source == "csv":
        return CsvLoader(cfg, **kwargs)
    if source == "newsgroups":
        return NewsgroupsLoader(cfg, **kwargs)
    raise DataError(f"unknown loader source: {source}", code="E200")
