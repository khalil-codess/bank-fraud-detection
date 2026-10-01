import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud.calibration import (
    CalibratedModel,
    brier_score,
    expected_calibration_error,
    fit_calibration,
    reliability,
)
from fraud.evaluate import decide


class _FixedScores:
    """Stand-in for a fitted pipeline: returns pre-computed scores."""

    named_steps = {"clf": None}

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, X):
        s = self.scores[np.asarray(X).ravel()]
        return np.column_stack([1 - s, s])


@pytest.fixture
def overconfident():
    """Real fraud risk p, but the model reports a much larger score (like class weighting does)."""
    rng = np.random.default_rng(0)
    true_p = rng.beta(0.3, 30, 20_000)
    y = (rng.random(20_000) < true_p).astype(int)
    raw = 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + 3)))  # inflated log-odds
    return y, raw


def test_calibration_fixes_overconfidence(overconfident):
    y, raw = overconfident
    model = fit_calibration(_FixedScores(raw), raw, y)
    calibrated = model.calibrate(raw)
    assert brier_score(y, calibrated) < brier_score(y, raw)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, raw) / 3
    assert calibrated.mean() == pytest.approx(y.mean(), rel=0.1)


def test_calibration_preserves_ranking(overconfident):
    y, raw = overconfident
    model = fit_calibration(_FixedScores(raw), raw, y)
    calibrated = model.predict_proba(np.arange(len(raw)))[:, 1]
    assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(calibrated, kind="stable"))
    assert average_precision_score(y, calibrated) == pytest.approx(average_precision_score(y, raw))


def test_negative_slope_is_refused():
    with pytest.raises(ValueError):
        CalibratedModel(None, slope=-1.0, intercept=0.0)


def test_reliability_bins_cover_every_row():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    rows = reliability((rng.random(1000) < p).astype(int), p, n_bins=10)
    assert sum(c for _, _, c in rows) == 1000
    assert all(0 <= obs <= 1 for _, obs, _ in rows)


def test_threshold_rule():
    policy = {"rule": "threshold", "threshold": 0.5}
    alerts = decide(np.array([0.1, 0.5, 0.9]), np.ones(3), policy)
    assert alerts.tolist() == [False, True, True]


def test_expected_value_rule_reviews_big_amounts_at_lower_risk():
    proba = np.array([0.05, 0.30, 0.30])
    amount = np.array([2_000.0, 10.0, 100.0])
    alerts = decide(proba, amount, {"rule": "expected_value", "review_cost": 5.0})
    # 0.05 x 2000 = 100 >= 5 -> review;  0.30 x 10 = 3 < 5 -> skip;  0.30 x 100 = 30 -> review
    assert alerts.tolist() == [True, False, True]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        decide(np.array([0.5]), np.array([1.0]), {"rule": "magic"})
