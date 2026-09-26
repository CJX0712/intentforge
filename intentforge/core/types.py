"""Shared data structures for IntentForge.

Every cross-module contract is expressed with these dataclasses so that
backends, the router and the evaluator speak the same language regardless of
the underlying library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextSample:
    """A single text instance."""

    text: str
    label: Optional[str] = None


@dataclass
class Prediction:
    """A single prediction produced by a backend."""

    label: str
    label_id: int
    confidence: float
    proba: Dict[str, float] = field(default_factory=dict)


@dataclass
class BackendResult:
    """All predictions of one backend over a dataset."""

    backend_name: str
    predictions: List[Prediction]


@dataclass
class RouterDecision:
    """Decision taken by the C3R router for one sample."""

    sample_index: int
    chosen_backend: str
    confidence: float
    abstain: bool
    reason: str


@dataclass
class BenchmarkReport:
    """Full benchmark artifact persisted as ``benchmark.json``."""

    environment: Dict[str, Any]
    dataset: Dict[str, Any]
    per_backend: Dict[str, Dict[str, Any]]
    cascade: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)
