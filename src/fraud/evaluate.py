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


def cost_metrics(y_true, proba, amount, threshold: float, review_cost: float) -> dict:
    y_true, amount = np.asarray(y_true), np.asarray(amount, dtype=float)
    alert = np.asarray(proba) >= threshold
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


def evaluate(y_true, proba, threshold: float, amount=None, review_cost: float | None = None) -> dict:
    """Metrics at the threshold, threshold-free ones (ROC-AUC, PR-AUC) and, given amounts, savings."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if amount is not None and review_cost is not None:
        out |= cost_metrics(y_true, proba, amount, threshold, review_cost)
    return out


def at_k(y_true, proba, k: int) -> dict:
    """Precision and recall when analysts can only review the k highest-scored transactions."""
    y_true = np.asarray(y_true)
    k = int(min(max(k, 1), len(y_true)))
    top = np.argsort(-np.asarray(proba), kind="stable")[:k]
    hits = int(y_true[top].sum())
    return {"k": k, "precision": hits / k, "recall": hits / max(int(y_true.sum()), 1), "frauds_caught": hits}


def bootstrap_ci(y_true, proba, metric=average_precision_score, n: int = 300, seed: int = 0):
    """95% bootstrap confidence interval (few frauds -> noisy metrics)."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_bootstrap(y_true, proba_a, proba_b, metric=average_precision_score,
                     n: int = 1000, seed: int = 0) -> dict:
    """Is model A better than model B on the same rows? Resamples rows jointly for both models."""
    rng = np.random.default_rng(seed)
    y_true, proba_a, proba_b = (np.asarray(x) for x in (y_true, proba_a, proba_b))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        diffs.append(metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff": float(metric(y_true, proba_a) - metric(y_true, proba_b)),
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_a_better": float((diffs > 0).mean()),
    }
