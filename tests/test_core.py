"""Tests for core errors/config."""

import os
from intentforge.core import errors
from intentforge.core.config import Config
from intentforge.core.errors import (
    BackendImportError,
    ConfigError,
    IntentForgeError,
    PipelineError,
)


def test_error_codes():
    assert IntentForgeError("x").code == "E000"
    assert ConfigError("x").code == "E100"
    assert BackendImportError("x").code == "E301"
    assert PipelineError("x").code == "E500"
    assert "E301" in str(BackendImportError("boom"))


def test_config_defaults():
    cfg = Config()
    assert cfg.random_seed == 42
    assert cfg.max_features == 4000
    assert cfg.ngram_tuple() == (1, 2)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("INTENTFORGE_RANDOM_SEED", "7")
    monkeypatch.setenv("INTENTFORGE_USE_XGBOOST", "false")
    monkeypatch.setenv("INTENTFORGE_MAX_FEATURES", "1234")
    cfg = Config.from_env()
    assert cfg.random_seed == 7
    assert cfg.use_xgboost is False
    assert cfg.max_features == 1234


def test_config_env_invalid(monkeypatch):
    monkeypatch.setenv("INTENTFORGE_RANDOM_SEED", "notanint")
    try:
        Config.from_env()
        assert False, "expected ConfigError"
    except ConfigError:
        pass
