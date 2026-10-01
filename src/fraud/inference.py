"""Loading trained artifacts, scoring and explaining transactions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from fraud.features import add_features


def load_artifacts(artifacts_dir: str | Path = "artifacts"):
    """Return (model, metrics). Only load model files you trained yourself: joblib runs code."""
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / "fraud_model.joblib")
    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, metrics


def score_transactions(model, threshold: float, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Score raw transactions -> DataFrame with fraud_proba and is_fraud."""
    proba = model.predict_proba(add_features(raw_df))[:, 1]
    return pd.DataFrame({"fraud_proba": proba, "is_fraud": proba >= threshold}, index=raw_df.index)


def explain(model, raw_df: pd.DataFrame):
    """SHAP explanation of the fraud class for a tree-based model (Random Forest, XGBoost)."""
    import shap

    X_t = model.named_steps["prep"].transform(add_features(raw_df))
    expl = shap.TreeExplainer(model.named_steps["clf"])(X_t)
    if expl.values.ndim == 3:  # Random Forest: one output per class
        expl = expl[:, :, 1]
    return expl
