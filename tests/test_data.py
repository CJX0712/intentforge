"""Tests for data generation / loading."""

from intentforge.core.config import Config
from intentforge.data.generator import generate_texts, make_splits
from intentforge.data.loaders import BuiltinLoader, get_loader


def _small_cfg():
    cfg = Config()
    cfg.n_samples = 300
    cfg.n_classes = 5
    cfg.test_size = 0.25
    cfg.val_size = 0.15
    return cfg


def test_generate_deterministic():
    cfg = _small_cfg()
    t1, l1 = generate_texts(cfg)
    t2, l2 = generate_texts(cfg)
    assert t1 == t2 and l1 == l2
    assert len(t1) == len(l1) == cfg.n_samples


def test_splits_shapes():
    cfg = _small_cfg()
    texts, labels = generate_texts(cfg)
    s = make_splits(texts, labels, cfg)
    assert set(s.keys()) >= {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "labels"}
    # every sample lands in exactly one of train / val / test
    total = len(s["X_train"]) + len(s["X_val"]) + len(s["X_test"])
    assert total == len(texts)
    assert len(s["X_train"]) > len(s["X_val"]) > 0
    # every label in train appears in label set
    assert set(s["y_train"]).issubset(set(s["labels"]))


def test_builtin_loader_runs():
    cfg = _small_cfg()
    loader = get_loader(cfg, "builtin")
    s = loader.load()
    assert len(s["X_test"]) > 0
    assert len(s["labels"]) == cfg.n_classes
