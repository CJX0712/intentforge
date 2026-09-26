"""C3R — Confidence-Calibrated Cascaded Router (system innovation).

Why it exists
-------------
Running a strong model on every request wastes latency/compute on "easy"
samples a cheap model already gets right. C3R routes **per sample**:

1. Calibrate each backend's raw ``max(proba)`` into an *empirical accuracy*
   via isotonic regression on a held-out calibration set. Calibrated scores
   are honest probabilities, so thresholds are interpretable.
2. If the fast backend's calibrated confidence ``>= tau_high`` -> use it.
3. Else if the strong backend's calibrated confidence ``>= tau_low`` ->
   escalate to it (cascade).
4. Else -> **abstain** (uncertainty band), handing the sample to a human /
   fallback instead of guessing.

Thresholds are tuned by Optuna (see :mod:`intentforge.hpo.tune`) to maximize
macro-F1 under a latency budget. Every step is independently unit-tested.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from intentforge.core.errors import RouterCalibrationError, RouterRouteError
from intentforge.core.types import Prediction, RouterDecision
from intentforge.core.interfaces import ITextBackend


class C3RCascadeRouter:
    """Confidence-calibrated cascaded router."""

    name = "c3r"

    def __init__(
        self,
        fast_backend: str = "sklearn_linear",
        strong_backend: str = "xgboost",
        tau_high: float = 0.80,
        tau_low: float = 0.55,
    ):
        self.fast_name = fast_backend
        self.strong_name = strong_backend
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.calibrators: Dict[str, object] = {}
        self._fitted = False

    # --- calibration --------------------------------------------------------
    def fit(self, backends: Dict[str, ITextBackend], texts: List[str], labels: List[str]) -> None:
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import roc_auc_score as _roc  # noqa: F401 (smoke import)
        except Exception as exc:
            raise RouterCalibrationError(
                "isotonic/metrics unavailable", code="E401", cause=exc
            ) from exc

        used = [n for n in (self.fast_name, self.strong_name) if n in backends]
        if not used:
            raise RouterCalibrationError(
                "no usable backend for router", code="E401"
            )

        for name in used:
            try:
                preds = backends[name].predict(texts)
            except Exception as exc:
                raise RouterCalibrationError(
                    f"backend '{name}' failed on calibration set: {exc}",
                    code="E401",
                    cause=exc,
                ) from exc
            scores = np.array([p.confidence for p in preds], dtype=float)
            correct = np.array(
                [1.0 if p.label == lab else 0.0 for p, lab in zip(preds, labels)],
                dtype=float,
            )
            if len(np.unique(scores)) < 2 or len(np.unique(correct)) < 2:
                # degenerate calibration set -> identity mapper
                self.calibrators[name] = _IdentityCalibrator()
            else:
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip"
                )
                iso.fit(scores, correct)
                self.calibrators[name] = iso
        self._fitted = True

    # --- routing ------------------------------------------------------------
    def _calibrate(self, name: str, raw: float) -> float:
        cal = self.calibrators.get(name)
        if cal is None:
            return float(raw)
        try:
            return float(cal.predict(np.array([raw], dtype=float))[0])
        except Exception:
            return float(raw)

    def route(self, sample_index: int, backend_outputs: Dict[str, Prediction]) -> RouterDecision:
        if not self._fitted:
            raise RouterRouteError("router not fitted", code="E402")
        cands: Dict[str, tuple] = {}
        for name, pred in backend_outputs.items():
            cands[name] = (pred, self._calibrate(name, pred.confidence))

        fast = cands.get(self.fast_name)
        strong = cands.get(self.strong_name)

        try:
            if fast is not None and fast[1] >= self.tau_high:
                return RouterDecision(
                    sample_index, self.fast_name, fast[1], False, "fast_confident"
                )
            if strong is not None and strong[1] >= self.tau_low:
                return RouterDecision(
                    sample_index, self.strong_name, strong[1], False, "escalated_to_strong"
                )
            if not cands:
                return RouterDecision(sample_index, "none", 0.0, True, "no_backend")
            best_name = max(cands, key=lambda n: cands[n][1])
            return RouterDecision(
                sample_index, best_name, cands[best_name][1], True, "abstain_low_conf"
            )
        except Exception as exc:
            raise RouterRouteError(
                f"routing failed: {exc}", code="E402", cause=exc
            ) from exc

    # --- lazy (true) cascade API -------------------------------------------
    def stage1(self, fast_pred: Prediction):
        """First stage: is the fast backend good enough?

        Returns ``(is_confident, calibrated_confidence)``. The strong backend
        must **not** be evaluated until this returns False — that is what makes
        the cascade cheaper than always running the strong model.
        """
        cal = self._calibrate(self.fast_name, fast_pred.confidence)
        return cal >= self.tau_high, cal

    def stage2(
        self, sample_index: int, fast_pred: Prediction, strong_pred: Prediction
    ) -> RouterDecision:
        """Second stage: decide between escalating and abstaining."""
        cf = self._calibrate(self.fast_name, fast_pred.confidence)
        cs = self._calibrate(self.strong_name, strong_pred.confidence)
        if cs >= self.tau_low:
            return RouterDecision(
                sample_index, self.strong_name, cs, False, "escalated_to_strong"
            )
        # both uncertain -> abstain, still report the better of the two
        if cs >= cf:
            return RouterDecision(sample_index, self.strong_name, cs, True, "abstain_low_conf")
        return RouterDecision(sample_index, self.fast_name, cf, True, "abstain_low_conf")

    # --- diagnostics --------------------------------------------------------
    def calibration_error(self, backends, texts, labels) -> Dict[str, float]:
        """Mean absolute calibration error per backend (lower = better)."""
        out: Dict[str, float] = {}
        for name in (self.fast_name, self.strong_name):
            if name not in backends:
                continue
            preds = backends[name].predict(texts)
            errs = []
            for p, lab in zip(preds, labels):
                cal = self._calibrate(name, p.confidence)
                errs.append(abs(cal - (1.0 if p.label == lab else 0.0)))
            out[name] = float(np.mean(errs)) if errs else 0.0
        return out


class _IdentityCalibrator:
    """Fallback calibrator used when the calibration set is degenerate."""

    def predict(self, x):
        return np.asarray(x, dtype=float)
