"""Evaluation: quality metrics + latency/memory profiling.

Sklearn metric imports are aliased (``_x``) because a same-named local helper
would shadow them and cause infinite recursion — a subtle bug we hit during
development and now guard against by convention.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List

from intentforge.core.errors import PipelineError

# aliased to avoid any accidental shadowing / recursion
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import f1_score as _f1_score


def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Accuracy + macro/weighted F1. Empty input -> zeros (safe)."""
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    try:
        acc = float(_accuracy_score(y_true, y_pred))
        macro = float(_f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted = float(_f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception as exc:
        raise PipelineError(f"metric computation failed: {exc}", code="E500", cause=exc) from exc
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def latency_profile(
    func: Callable[[List[str]], List], texts: List[str], runs: int = 3
) -> Dict[str, float]:
    """Time ``func`` over ``texts`` for ``runs`` repetitions.

    Returns per-sample p50/p95/mean latency (ms) and throughput (samples/s).
    """
    if not texts:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "throughput_per_s": 0.0}
    all_durations: List[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        func(texts)
        t1 = time.perf_counter()
        all_durations.append((t1 - t0) * 1000.0)

    total_ms = sum(all_durations) / len(all_durations)
    per_sample = total_ms / len(texts)
    # crude p50/p95 from repetitions (enough for a baseline; not a distribution)
    ordered = sorted(all_durations)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "p50_ms": round(p50 / len(texts), 4),
        "p95_ms": round(p95 / len(texts), 4),
        "mean_ms": round(per_sample, 4),
        "throughput_per_s": round(1000.0 / per_sample, 2) if per_sample > 0 else 0.0,
    }


def memory_rss_mb() -> float:
    """Resident set size in MB, or 0.0 if psutil unavailable."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0
