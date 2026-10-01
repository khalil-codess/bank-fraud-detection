"""Hyper-parameter search with Optuna, inside the training period only.

Usage: python -m fraud.tune --config configs/sparkov.yaml --model LightGBM --trials 40

Protocol (validation and test are never touched):
  - the chronological training window is split again: fit on its first 80%, score on its last 20%;
  - legitimate rows of the fit part can be down-sampled (--negatives) to make trials fast; the
    score is PR-AUC on the untouched recent 20%, which down-sampling does not bias;
  - the current config settings are scored on the same split, so the gain is measured honestly.

The result is written to outputs/<dataset>/tuning_<model>.json. Copy `best_params` into the
config by hand: the config stays the single source of truth for training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.config import load_config
from fraud.data import load_transactions, time_split
from fraud.datasets import get_dataset
from fraud.models import build_models

log = logging.getLogger("fraud.tune")


def search_space(model: str, trial) -> dict:
    """Parameters Optuna explores for each tunable model."""
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if model == "LightGBM":
        return common | {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample_freq": 1,
        }
    if model in ("XGBoost", "XGBoost + SMOTE"):
        return common | {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        }
    raise ValueError(f"No search space for {model!r}; tunable: LightGBM, XGBoost, XGBoost + SMOTE")


def inner_split(train: pd.DataFrame, negatives: float, seed: int):
    """Fit on the older 80% of the training window (legitimate rows down-sampled), score on the
    most recent 20%."""
    cut = int(len(train) * 0.8)
    fit, holdout = train.iloc[:cut], train.iloc[cut:]
    if negatives < 1:
        legit = fit[fit["Class"] == 0].sample(frac=negatives, random_state=seed)
        fit = pd.concat([fit[fit["Class"] == 1], legit]).sort_values("Time", kind="stable")
    return fit, holdout


def tune(cfg, model: str, trials: int, negatives: float, timeout: float | None = None) -> dict:
    import optuna

    spec = get_dataset(cfg.dataset)
    train, _, _ = time_split(load_transactions(spec, cfg.data_path), cfg.val_frac, cfg.test_frac)
    fit, holdout = inner_split(train, negatives, cfg.seed)
    X_fit, X_hold = spec.add_features(fit), spec.add_features(holdout)
    log.info("Tuning %s on %s fit rows (%d frauds), scored on %s recent rows (%d frauds)", model,
             f"{len(fit):,}", int(fit["Class"].sum()), f"{len(holdout):,}", int(holdout["Class"].sum()))

    def score(params: dict) -> float:
        pipe = build_models(fit["Class"], {model: params}, spec.scale_cols, seed=cfg.seed)[model]
        pipe.fit(X_fit, fit["Class"])
        return float(average_precision_score(holdout["Class"], pipe.predict_proba(X_hold)[:, 1]))

    current = dict(cfg.models.get(model) or {})
    baseline = score(dict(current))
    log.info("Current config settings: PR-AUC %.4f", baseline)

    extra = {k: v for k, v in current.items() if k == "smote_ratio"}  # not searched, kept as is
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    start = time.time()

    def objective(trial):
        value = score(search_space(model, trial) | extra)
        log.info("trial %3d  PR-AUC %.4f", trial.number, value)
        return value

    study.optimize(objective, n_trials=trials, timeout=timeout)
    best_params = search_space(model, optuna.trial.FixedTrial(study.best_params)) | extra
    result = {
        "dataset": cfg.dataset, "model": model, "trials": len(study.trials),
        "seconds": round(time.time() - start), "negatives_kept": negatives,
        "fit_rows": len(fit), "holdout_rows": len(holdout),
        "current_params": current, "current_pr_auc": baseline,
        "best_params": best_params, "best_pr_auc": study.best_value,
        "history": [t.value for t in study.trials],
    }
    out = cfg.outputs_dir / f"tuning_{model.replace(' ', '_').replace('+', 'plus')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Best PR-AUC %.4f vs current %.4f (%+.4f). Written to %s", study.best_value, baseline,
             study.best_value - baseline, out)
    log.info("best_params:\n%s", "\n".join(f"  {k}: {_fmt(v)}" for k, v in best_params.items()))
    return result


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/sparkov.yaml")
    ap.add_argument("--model", default="LightGBM")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=float, help="stop after this many seconds")
    ap.add_argument("--negatives", type=float, default=0.2,
                    help="share of legitimate fit rows kept (speed); frauds are always kept")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    np.seterr(all="ignore")
    tune(load_config(args.config), args.model, args.trials, args.negatives, args.timeout)


if __name__ == "__main__":
    main()
