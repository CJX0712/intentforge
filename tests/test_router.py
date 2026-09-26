"""Tests for the C3R router (the system innovation).

Routing logic is tested with an identity calibrator so thresholds map directly
onto raw confidences — deterministic and independent of ML libraries.
"""

import pytest

from intentforge.core.errors import RouterCalibrationError, RouterRouteError
from intentforge.core.types import Prediction
from intentforge.domain.router import C3RCascadeRouter, _IdentityCalibrator


def _router(tau_high=0.80, tau_low=0.55):
    r = C3RCascadeRouter(fast_backend="fast", strong_backend="strong",
                         tau_high=tau_high, tau_low=tau_low)
    r.calibrators = {"fast": _IdentityCalibrator(), "strong": _IdentityCalibrator()}
    r._fitted = True
    return r


def test_route_fast_confident():
    r = _router()
    d = r.route(0, {"fast": Prediction("a", 0, 0.90), "strong": Prediction("a", 0, 0.90)})
    assert d.chosen_backend == "fast"
    assert d.abstain is False
    assert d.reason == "fast_confident"


def test_route_escalates_when_fast_uncertain():
    r = _router()
    d = r.route(1, {"fast": Prediction("a", 0, 0.30), "strong": Prediction("a", 0, 0.90)})
    assert d.chosen_backend == "strong"
    assert d.abstain is False
    assert d.reason == "escalated_to_strong"


def test_route_abstains_when_both_uncertain():
    r = _router()
    d = r.route(2, {"fast": Prediction("a", 0, 0.20), "strong": Prediction("a", 0, 0.30)})
    assert d.abstain is True
    assert d.reason == "abstain_low_conf"
    assert d.chosen_backend == "strong"  # best available still reported


def test_route_single_backend_degrades_gracefully():
    r = C3RCascadeRouter("fast", "strong")
    r.calibrators = {"fast": _IdentityCalibrator()}
    r._fitted = True
    confident = r.route(0, {"fast": Prediction("a", 0, 0.95)})
    assert confident.chosen_backend == "fast" and confident.abstain is False
    weak = r.route(1, {"fast": Prediction("a", 0, 0.10)})
    assert weak.chosen_backend == "fast" and weak.abstain is True


def test_threshold_boundary_is_inclusive():
    r = _router(tau_high=0.80, tau_low=0.55)
    d = r.route(0, {"fast": Prediction("a", 0, 0.80), "strong": Prediction("a", 0, 0.10)})
    assert d.chosen_backend == "fast" and not d.abstain


def test_route_before_fit_raises():
    r = C3RCascadeRouter("fast", "strong")
    with pytest.raises(RouterRouteError):
        r.route(0, {"fast": Prediction("a", 0, 0.9)})


def test_calibration_without_backend_raises():
    r = C3RCascadeRouter("fast", "strong")
    from intentforge.core.errors import RouterCalibrationError

    with pytest.raises(RouterCalibrationError):
        r.fit({}, ["x"], ["a"])


def test_identity_calibrator_is_identity():
    import numpy as np

    cal = _IdentityCalibrator()
    assert float(cal.predict(np.array([0.42]))[0]) == 0.42
