"""Configuration with declarative ENV overrides.

All settings can be overridden via ``INTENTFORGE_<NAME>`` environment variables
so the same image runs unchanged across environments (12-factor style).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


@dataclass
class Config:
    """Top-level runtime configuration."""

    # reproducibility
    random_seed: int = 42

    # dataset
    n_samples: int = 2400
    n_classes: int = 10
    test_size: float = 0.25
    val_size: float = 0.15  # fraction of *train* held out for calibration

    # vectorization (shared by sklearn / xgboost backends)
    max_features: int = 4000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    sublinear_tf: bool = True

    # backends (toggle; availability is probed at runtime)
    use_sklearn_linear: bool = True
    use_xgboost: bool = True
    use_transformer: bool = False  # optional; off by default for offline reproducibility
    transformer_model: str = "distilbert-base-uncased"

    # router thresholds (calibrated confidence space, 0..1)
    tau_high: float = 0.80  # fast backend confident enough -> use directly
    tau_low: float = 0.55  # strong backend floor before abstaining

    # hpo
    hpo_enabled: bool = True
    hpo_trials: int = 25

    # benchmark
    benchmark_sample: int = 600  # number of test samples timed for latency
    latency_runs: int = 3

    # misc
    label_names: List[str] = field(
        default_factory=lambda: [
            "greeting", "query_balance", "transfer_money", "complaint",
            "cancel_subscription", "technical_support", "human_agent",
            "payment_issue", "account_access", "product_info",
        ]
    )

    # --- helpers -----------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, applying ``INTENTFORGE_*`` overrides."""
        cfg = cls()
        prefix = "INTENTFORGE_"
        for key in list(cls.__dataclass_fields__.keys()):  # type: ignore[attr-defined]
            env_key = prefix + key.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                cur = getattr(cfg, key)
                try:
                    if isinstance(cur, bool):
                        setattr(cfg, key, raw.lower() in ("1", "true", "yes", "on"))
                    elif isinstance(cur, int):
                        setattr(cfg, key, int(raw))
                    elif isinstance(cur, float):
                        setattr(cfg, key, float(raw))
                    elif isinstance(cur, list):
                        setattr(cfg, key, [s.strip() for s in raw.split(",") if s.strip()])
                    else:
                        setattr(cfg, key, raw)
                except (ValueError, TypeError) as exc:
                    from intentforge.core.errors import ConfigError

                    raise ConfigError(
                        f"invalid ENV {env_key}={raw!r}", code="E100", cause=exc
                    ) from exc
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ngram_tuple(self) -> Tuple[int, int]:
        return (self.ngram_min, self.ngram_max)
