"""Tests for backends."""

import pytest

from intentforge.core.config import Config
from intentforge.data.generator import generate_texts, make_splits
from intentforge.domain.backends import (
    XGBOOST_AVAILABLE,
    SklearnLinearBackend,
    XGBoostBackend,
    build_backends,
)


def _cfg(n=400, classes=5):
    cfg = Config()
    cfg.n_samples = n
    cfg.n_classes = classes
    cfg.test_size = 0.25
    cfg.val_size = 0.15
    return cfg


def _splits(cfg):
    texts, labels = generate_texts(cfg)
    return make_splits(texts, labels, cfg)


def test_sklearn_linear_backend():
    cfg = _cfg()
    s = _splits(cfg)
    b = SklearnLinearBackend(cfg)
    assert b.available() is True
    b.fit(s["X_train"], s["y_train"])
    preds = b.predict(s["X_test"])
    assert len(preds) == len(s["X_test"])
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0
        assert abs(sum(p.proba.values()) - 1.0) < 1e-5
        assert p.label in s["labels"]
    acc = sum(1 for p, y in zip(preds, s["y_test"]) if p.label == y) / len(preds)
    assert acc > 0.5, f"sanity accuracy too low: {acc}"


def test_sklearn_linear_predict_before_fit_errors():
    b = SklearnLinearBackend(_cfg())
    from intentforge.core.errors import BackendPredictError

    with pytest.raises(BackendPredictError):
        b.predict(["hello"])


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_xgboost_backend():
    cfg = _cfg()
    s = _splits(cfg)
    b = XGBoostBackend(cfg)
    assert b.available() is True
    b.fit(s["X_train"], s["y_train"])
    preds = b.predict(s["X_test"])
    assert len(preds) == len(s["X_test"])
    for p in preds:
        assert abs(sum(p.proba.values()) - 1.0) < 1e-5
    acc = sum(1 for p, y in zip(preds, s["y_test"]) if p.label == y) / len(preds)
    assert acc > 0.5, f"xgboost accuracy too low: {acc}"


def test_build_backends_contains_sklearn():
    backends = build_backends(_cfg())
    assert "sklearn_linear" in backends
    allb = list(backends.keys())
    assert len(allb) >= 1
