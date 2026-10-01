"""Rolling-origin (time-series) cross-validation.

Each fold trains on everything before a cut-off and tests on the next block of time:

    fold 1: [ fit ......... | calib ] [ test ]
    fold 2: [ fit ............. | calib ] [ test ]
    ...

`calib` (the most recent 20% of the history) picks the decision threshold, so every fold
mimics the real workflow: train on the past, set the threshold, score the future.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraud.evaluate import evaluate, select_threshold, select_threshold_by_cost
from fraud.features import add_features
from fraud.models import build_models

log = logging.getLogger(__name__)

SUMMARY_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "savings_rate")


def rolling_folds(df: pd.DataFrame, n_folds: int = 4, fold_frac: float = 0.1, calib_frac: float = 0.2):
    """Yield (fit, calib, test) DataFrames for each fold, in chronological order."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n, fold_size = len(df), int(len(df) * fold_frac)
    if n_folds * fold_size >= n:
        raise ValueError("n_folds * fold_frac must leave some data for the first training window")
    for i in range(n_folds):
        start = n - (n_folds - i) * fold_size
        history, test = df.iloc[:start], df.iloc[start:start + fold_size]
        cut = int(len(history) * (1 - calib_frac))
        yield history.iloc[:cut], history.iloc[cut:], test


def pick_threshold(y, proba, amount, cfg) -> float:
    if cfg.threshold_method == "cost":
        return select_threshold_by_cost(y, proba, amount, cfg.review_cost)
    return select_threshold(y, proba, beta=cfg.beta)


def cross_validate(df: pd.DataFrame, cfg) -> dict:
    """Per-model fold metrics plus mean/std summaries. `df` must exclude the final test set."""
    folds_out: dict[str, list] = {name: [] for name in cfg.models}
    for i, (fit, calib, test) in enumerate(rolling_folds(df, cfg.cv_folds, cfg.cv_fold_frac), 1):
        if min(fit["Class"].sum(), calib["Class"].sum(), test["Class"].sum()) == 0:
            log.warning("CV fold %d skipped: a window contains no fraud", i)
            continue
        log.info("CV fold %d/%d: fit %s, calib %s, test %s rows (%d frauds)", i, cfg.cv_folds,
                 f"{len(fit):,}", f"{len(calib):,}", f"{len(test):,}", int(test["Class"].sum()))
        models = build_models(fit["Class"], cfg.models, seed=cfg.seed)
        for name, model in models.items():
            model.fit(add_features(fit), fit["Class"])
            p_calib = model.predict_proba(add_features(calib))[:, 1]
            thr = pick_threshold(calib["Class"], p_calib, calib["Amount"], cfg)
            p_test = model.predict_proba(add_features(test))[:, 1]
            folds_out[name].append(evaluate(test["Class"], p_test, thr, test["Amount"], cfg.review_cost))

    return {name: _summarise(folds) for name, folds in folds_out.items()}


def _summarise(folds: list[dict]) -> dict:
    """Mean/std of each summary metric; empty when every fold was skipped."""
    if not folds:
        return {}
    summary = {"folds": folds}
    for m in SUMMARY_METRICS:
        values = [f[m] for f in folds]
        summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary
