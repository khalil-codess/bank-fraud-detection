"""Probability calibration and calibration metrics.

Class weights and SMOTE make models over-confident about fraud: a raw score of 0.3 can mean a 2%
real risk. `CalibratedModel` maps the score through a logistic fit on its log-odds (Platt scaling).
The map is strictly increasing, so the ranking, PR-AUC and SHAP explanations are unchanged; only
the numbers become probabilities you can reason with, e.g. "expected loss = probability × amount".
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# Clip only exact 0 and 1: tree models output scores far below 1e-7 for obvious legitimate
# transactions, and a coarser clip would tie them and lose their order.
_LOW, _HIGH = np.finfo(float).tiny, np.nextafter(1.0, 0.0)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _LOW, _HIGH)
    return np.log(p) - np.log1p(-p)


class CalibratedModel:
    """A fitted pipeline whose fraud scores pass through a monotone Platt calibration."""

    def __init__(self, pipeline, slope: float, intercept: float):
        if slope <= 0:
            raise ValueError("calibration slope must be positive to preserve the ranking")
        self.pipeline, self.slope, self.intercept = pipeline, float(slope), float(intercept)

    @property
    def named_steps(self):  # SHAP explains the underlying model's raw score
        return self.pipeline.named_steps

    def calibrate(self, raw_proba):
        return expit(self.slope * _logit(raw_proba) + self.intercept)

    def predict_proba(self, X):
        q = self.calibrate(self.pipeline.predict_proba(X)[:, 1])
        return np.column_stack([1 - q, q])


def fit_calibration(pipeline, raw_proba, y_true) -> CalibratedModel:
    """Fit Platt scaling on held-out scores (validation data, never the test set)."""
    lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularised
    lr.fit(_logit(raw_proba).reshape(-1, 1), np.asarray(y_true))
    return CalibratedModel(pipeline, lr.coef_[0, 0], lr.intercept_[0])


def brier_score(y_true, proba) -> float:
    return float(np.mean((np.asarray(proba, dtype=float) - np.asarray(y_true)) ** 2))


def reliability(y_true, proba, n_bins: int = 10):
    """Equal-width bins of predicted probability -> (mean predicted, observed fraud rate, count)
    for every non-empty bin."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    idx = np.clip((proba * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((float(proba[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return rows


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Count-weighted gap between predicted probability and observed fraud rate."""
    rows = reliability(y_true, proba, n_bins)
    total = sum(c for _, _, c in rows)
    return float(sum(c * abs(pred - obs) for pred, obs, c in rows) / total)
