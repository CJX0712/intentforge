"""Tests for evaluation helpers."""

from intentforge.eval.metrics import (
    classification_metrics,
    latency_profile,
    memory_rss_mb,
)


def test_classification_metrics_values():
    y = ["a", "b", "a", "b"]
    p = ["a", "b", "b", "b"]
    m = classification_metrics(y, p)
    assert abs(m["accuracy"] - 0.75) < 1e-6
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["weighted_f1"] <= 1.0


def test_perfect_prediction():
    y = ["a", "b", "a"]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_empty_input_is_safe():
    m = classification_metrics([], [])
    assert m["accuracy"] == 0.0 and m["macro_f1"] == 0.0


def test_latency_profile_returns_shape():
    prof = latency_profile(lambda texts: texts, ["hello"] * 20, runs=3)
    for k in ("p50_ms", "p95_ms", "mean_ms", "throughput_per_s"):
        assert k in prof
    assert prof["mean_ms"] >= 0.0


def test_latency_profile_empty_is_safe():
    prof = latency_profile(lambda texts: texts, [], runs=1)
    assert prof["mean_ms"] == 0.0


def test_memory_rss_non_negative():
    assert memory_rss_mb() >= 0.0
