"""Protocol contracts for every swappable component.

Implementations live in ``domain`` / ``data`` / ``eval``; the pipeline only
depends on these protocols, which keeps modules independently testable.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from intentforge.core.types import (
    BackendResult,
    Prediction,
    RouterDecision,
    TextSample,
)


@runtime_checkable
class ITextBackend(Protocol):
    """A text classifier backend."""

    name: str

    def available(self) -> bool:
        """Return True if the underlying library is importable."""
        ...

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Train on raw texts + string labels."""
        ...

    def predict(self, texts: List[str]) -> List[Prediction]:
        """Predict labels + probabilities for each text."""
        ...


@runtime_checkable
class IRouter(Protocol):
    """Routes each sample to the most appropriate backend."""

    def fit(self, backends: Dict[str, ITextBackend], texts: List[str], labels: List[str]) -> None:
        ...

    def route(
        self, sample_index: int, backend_outputs: Dict[str, Prediction]
    ) -> RouterDecision:
        ...


@runtime_checkable
class IDataLoader(Protocol):
    """Loads a train/test split with a stable label order."""

    def load(self) -> Dict[str, Any]:
        """Return dict with X_train, y_train, X_test, y_test, labels."""
        ...


@runtime_checkable
class IEvaluator(Protocol):
    """Computes quality + performance metrics."""

    def classification_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        ...

    def latency_profile(self, texts: List[str]) -> Dict[str, float]:
        ...
