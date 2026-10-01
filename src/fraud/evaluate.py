"""Thresholds, statistical and business metrics.

Cost model used throughout:
  - a missed fraud costs its transaction amount;
  - every alert (true or false) costs `review_cost` to investigate;
  - savings = fraud amount caught - review_cost * number of alerts.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true, proba, beta: float = 1.0) -> float:
    """Threshold maximising F-beta (beta > 1 favours recall: missing a fraud is costly)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    fbeta = np.divide((1 + beta**2) * precision * recall, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def savings_curve(y_true, proba, amount, review_cost: float):
    """Savings for every distinct threshold, from the strictest to the most lenient.

    Returns (thresholds, n_alerts, savings) where alerts are transactions with proba >= threshold.
    """
    y_true, proba, amount = (np.asarray(x, dtype=float) for x in (y_true, proba, amount))
    order = np.argsort(-proba, kind="stable")
    p = proba[order]
    caught = np.cumsum(amount[order] * y_true[order])
    n_alerts = np.arange(1, len(p) + 1)
    last_of_tie = np.r_[p[1:] != p[:-1], True]  # a threshold flags every tied score at once
    return p[last_of_tie], n_alerts[last_of_tie], (caught - review_cost * n_alerts)[last_of_tie]


def select_threshold_by_cost(y_true, proba, amount, review_cost: float) -> float:
    """Threshold maximising savings. If no threshold saves money, flag nothing."""
    thresholds, _, savings = savings_curve(y_true, proba, amount, review_cost)
    i = int(np.argmax(savings))
    if savings[i] <= 0:
        return float(np.nextafter(thresholds[0], np.inf))
    return float(thresholds[i])


def decide(proba, amount, policy: dict) -> np.ndarray:
    """Which transactions to send for review under a decision policy.

    - {"rule": "threshold", "threshold": t}: alert when the score >= t.
    - {"rule": "expected_value", "review_cost": c}: alert when probability x amount >= c, i.e. when
      the expected fraud loss avoided pays for the review. Needs calibrated probabilities, and
      reviews a large purchase at a lower risk than a small one.
    """
    proba = np.asarray(proba, dtype=float)
    if policy["rule"] == "threshold":
        return proba >= policy["threshold"]
    if policy["rule"] == "expected_value":
        return proba * np.asarray(amount, dtype=float) >= policy["review_cost"]
    raise ValueError(f"Unknown decision rule {policy['rule']!r}")


def cost_metrics(y_true, alert, amount, review_cost: float) -> dict:
    y_true, amount, alert = np.asarray(y_true), np.asarray(amount, dtype=float), np.asarray(alert, bool)
    fraud_total = float(amount[y_true == 1].sum())
    caught = float(amount[(y_true == 1) & alert].sum())
    review = float(review_cost * alert.sum())
    return {
        "fraud_amount_total": fraud_total,
        "fraud_amount_caught": caught,
        "review_cost_total": review,
        "savings": caught - review,
        # share of the fraud losses the model avoids (net of review costs); comparable across periods
        "savings_rate": (caught - review) / fraud_total if fraud_total > 0 else 0.0,
        "n_alerts": int(alert.sum()),
    }


def evaluate(y_true, proba, threshold: float | None = None, amount=None,
             review_cost: float | None = None, alerts=None) -> dict:
    """Metrics of the alerts (`alerts`, or score >= `threshold`), threshold-free ones (ROC-AUC,
    PR-AUC) and, given amounts and a review cost, the savings."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold if alerts is None else np.asarray(alerts, bool)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": None if threshold is None else float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if amount is not None and review_cost is not None:
        out |= cost_metrics(y_true, pred, amount, review_cost)
    return out


def at_k(y_true, proba, k: int) -> dict:
    """Precision and recall when analysts can only review the k highest-scored transactions."""
    y_true = np.asarray(y_true)
    k = int(min(max(k, 1), len(y_true)))
    top = np.argsort(-np.asarray(proba), kind="stable")[:k]
    hits = int(y_true[top].sum())
    return {"k": k, "precision": hits / k, "recall": hits / max(int(y_true.sum()), 1), "frauds_caught": hits}


class _SortedScores:
    """Scores sorted once, so average precision can be recomputed for any row weights in O(n).

    A bootstrap resample is equivalent to weighting each row by how many times it was drawn, so
    resampling never needs to re-sort. Matches sklearn's average_precision_score, ties included.
    """

    def __init__(self, y_true, proba):
        order = np.argsort(-np.asarray(proba, dtype=float), kind="stable")
        self.order = order
        self.y = np.asarray(y_true, dtype=float)[order]
        p = np.asarray(proba, dtype=float)[order]
        self.tie_last = np.r_[p[1:] != p[:-1], True]  # one threshold per distinct score

    def average_precision(self, weights=None) -> float:
        w = np.ones_like(self.y) if weights is None else np.asarray(weights, dtype=float)[self.order]
        tp = np.cumsum(w * self.y)[self.tie_last]
        fp = np.cumsum(w * (1 - self.y))[self.tie_last]
        if tp[-1] == 0:
            return float("nan")
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _bootstrap_weights(n_rows: int, n: int, seed: int):
    """Yield row weights for `n` bootstrap resamples (count of each row in the resample)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows)


def bootstrap_ci(y_true, proba, n: int = 300, seed: int = 0):
    """95% bootstrap confidence interval of PR-AUC (few frauds -> noisy metrics)."""
    scores = _SortedScores(y_true, proba)
    values = [scores.average_precision(w) for w in _bootstrap_weights(len(scores.y), n, seed)]
    values = np.asarray(values)[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_bootstrap(y_true, proba_a, proba_b, n: int = 1000, seed: int = 0) -> dict:
    """Is model A's PR-AUC better than model B's on the same rows? Both see identical resamples."""
    a, b = _SortedScores(y_true, proba_a), _SortedScores(y_true, proba_b)
    diffs = np.asarray([a.average_precision(w) - b.average_precision(w)
                        for w in _bootstrap_weights(len(a.y), n, seed)])
    diffs = diffs[~np.isnan(diffs)]
    return {
        "diff": a.average_precision() - b.average_precision(),
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_a_better": float((diffs > 0).mean()),
    }
