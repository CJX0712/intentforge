"""End-to-end integration tests for the pipeline."""

import json

from intentforge.core.config import Config
from intentforge.core.errors import PipelineError
from intentforge.domain.backends import XGBOOST_AVAILABLE
from intentforge.pipeline.pipeline import IntentForgePipeline


def _cfg():
    cfg = Config()
    cfg.n_samples = 700
    cfg.n_classes = 6
    cfg.test_size = 0.25
    cfg.val_size = 0.15
    cfg.hpo_trials = 4
    cfg.benchmark_sample = 80
    cfg.latency_runs = 2
    return cfg


def test_train_evaluate_smoke():
    pipe = IntentForgePipeline(_cfg())
    pipe.train()
    ev = pipe.evaluate()
    assert "sklearn_linear" in ev["per_backend"]
    cascade = ev["cascade"]
    for k in ("accuracy", "macro_f1", "frac_fast", "frac_strong", "frac_abstain"):
        assert k in cascade
    assert 0.0 <= cascade["accuracy"] <= 1.0
    # cascade must actually produce predictions (routing happened)
    assert cascade["frac_fast"] + cascade["frac_strong"] > 0.0


def test_cascade_not_worse_than_linear():
    """Core innovation claim: cascade >= fast backend (within tolerance)."""
    if not XGBOOST_AVAILABLE:
        import pytest

        pytest.skip("cascade comparison needs xgboost")
    pipe = IntentForgePipeline(_cfg())
    pipe.train()
    ev = pipe.evaluate()
    lin = ev["per_backend"]["sklearn_linear"]["macro_f1"]
    casc = ev["cascade"]["macro_f1"]
    assert casc >= lin - 0.06, f"cascade {casc:.4f} vs linear {lin:.4f}"


def test_benchmark_writes_evidence(tmp_path):
    pipe = IntentForgePipeline(_cfg())
    pipe.train()
    pipe.evaluate()
    rep = pipe.benchmark()
    assert rep.per_backend["sklearn_linear"]["mean_ms"] > 0.0
    assert rep.cascade["mean_ms"] > 0.0
    assert rep.environment["python"]
    assert rep.dataset["n_test"] > 0

    out = tmp_path / "benchmark.json"
    pipe.save_benchmark(str(out))
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("environment", "dataset", "per_backend", "cascade", "timestamp"):
        assert key in data


def test_predict_returns_routing_decisions():
    pipe = IntentForgePipeline(_cfg())
    pipe.train()
    res = pipe.predict(["please check my balance", "cancel my subscription now"])
    assert len(res) == 2
    for r in res:
        assert "label" in r and "backend" in r and "calibrated_confidence" in r
        assert isinstance(r["abstain"], bool)


def test_evaluate_before_train_raises():
    pipe = IntentForgePipeline(_cfg())
    try:
        pipe.evaluate()
        assert False, "expected PipelineError"
    except PipelineError:
        pass
