"""Hyper-parameter optimization for the router thresholds via Optuna.

Tunes ``tau_high`` / ``tau_low`` to maximize macro-F1 while keeping a healthy
fraction of samples on the fast backend (which is what buys the latency win).
"""

from __future__ import annotations

from typing import Dict, List

from intentforge.core.errors import ConfigError
from intentforge.domain.cascade import cascade_predict
from intentforge.eval.metrics import classification_metrics


def tune_router(router, backends, texts: List[str], labels: List[str], n_trials: int = 25) -> Dict[str, float]:
    """Optimize ``tau_high``/``tau_low`` in place; return best params."""
    try:
        import optuna
    except Exception as exc:
        raise ConfigError(
            "optuna required for HPO (pip install optuna)", code="E100", cause=exc
        ) from exc

    def objective(trial):
        th = trial.suggest_float("tau_high", 0.50, 0.99)
        tl = trial.suggest_float("tau_low", 0.30, 0.95)
        router.tau_high = th
        router.tau_low = tl
        preds, decisions = cascade_predict(router, backends, texts)
        y_pred = [p.label for p in preds]
        m = classification_metrics(labels, y_pred)
        n = max(1, len(decisions))
        frac_fast = sum(
            1 for d in decisions if d.chosen_backend == router.fast_name and not d.abstain
        ) / n
        penalty = 0.0 if frac_fast >= 0.30 else (0.30 - frac_fast) * 0.5
        return m["macro_f1"] - penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max(1, n_trials))

    best = study.best_params
    router.tau_high = best["tau_high"]
    router.tau_low = best["tau_low"]
    return {
        "tau_high": router.tau_high,
        "tau_low": router.tau_low,
        "best_objective": float(study.best_value),
    }
