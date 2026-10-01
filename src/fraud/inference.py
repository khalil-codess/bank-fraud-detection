"""Loading trained artifacts, scoring and explaining transactions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.datasets import DatasetSpec, get_dataset
from fraud.evaluate import decide


def load_artifacts(artifacts_dir: str | Path):
    """Return (model, metrics, dataset spec). Only load model files you trained yourself: joblib runs code."""
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / "fraud_model.joblib")
    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, metrics, get_dataset(metrics["dataset"])


def score_transactions(model, policy: dict, raw_df: pd.DataFrame, spec: DatasetSpec,
                       features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score raw transactions -> fraud_proba, expected_loss (= proba x amount) and is_fraud (alert
    under the decision policy saved in metrics.json). Pass `features` to reuse
    spec.add_features(raw_df) already computed by the caller."""
    proba = model.predict_proba(spec.add_features(raw_df) if features is None else features)[:, 1]
    return pd.DataFrame({
        "fraud_proba": proba,
        "expected_loss": proba * raw_df["Amount"].to_numpy(dtype=float),
        "is_fraud": decide(proba, raw_df["Amount"], policy),
    }, index=raw_df.index)


_EXPLAINERS: dict = {}


def _explainer(clf):
    """One SHAP TreeExplainer per fitted model (building it costs far more than using it)."""
    import shap

    key = id(clf)
    if key not in _EXPLAINERS or _EXPLAINERS[key][0] is not clf:
        _EXPLAINERS[key] = (clf, shap.TreeExplainer(clf))
    return _EXPLAINERS[key][1]


def explain(model, raw_df: pd.DataFrame, spec: DatasetSpec, features: pd.DataFrame | None = None):
    """Additive explanation of the fraud score in log-odds: SHAP for tree models (Random Forest,
    XGBoost, LightGBM); coefficient x value for linear models (exact, baseline = all-zero input)."""
    import shap

    X_t = model.named_steps["prep"].transform(spec.add_features(raw_df) if features is None else features)
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = X_t.to_numpy(dtype=float) * clf.coef_[0]
        return shap.Explanation(values=values, base_values=np.full(len(X_t), clf.intercept_[0]),
                                data=X_t.to_numpy(), feature_names=list(X_t.columns))
    expl = _explainer(clf)(X_t)
    if expl.values.ndim == 3:  # Random Forest: one output per class
        expl = expl[:, :, 1]
    return expl
