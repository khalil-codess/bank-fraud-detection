"""Threshold selection and evaluation metrics."""

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


def evaluate(y_true, proba, threshold: float) -> dict:
    """Metrics at the chosen threshold plus threshold-free ones (ROC-AUC, PR-AUC)."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


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
