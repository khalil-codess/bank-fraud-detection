"""Loading trained artifacts, scoring and explaining transactions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from fraud.datasets import DatasetSpec, get_dataset
from fraud.evaluate import decide


def load_artifacts(artifacts_dir: str | Path):
    """Return (model, metrics, dataset spec). Only load model files you trained yourself: joblib runs code."""
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / "fraud_model.joblib")
    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, metrics, get_dataset(metrics["dataset"])


def score_transactions(model, policy: dict, raw_df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Score raw transactions -> fraud_proba, expected_loss (= proba x amount) and is_fraud (alert
    under the decision policy saved in metrics.json)."""
    proba = model.predict_proba(spec.add_features(raw_df))[:, 1]
    return pd.DataFrame({
        "fraud_proba": proba,
        "expected_loss": proba * raw_df["Amount"].to_numpy(dtype=float),
        "is_fraud": decide(proba, raw_df["Amount"], policy),
    }, index=raw_df.index)


def explain(model, raw_df: pd.DataFrame, spec: DatasetSpec):
    """SHAP explanation of the fraud class for a tree-based model (Random Forest, XGBoost)."""
    import shap

    X_t = model.named_steps["prep"].transform(spec.add_features(raw_df))
    expl = shap.TreeExplainer(model.named_steps["clf"])(X_t)
    if expl.values.ndim == 3:  # Random Forest: one output per class
        expl = expl[:, :, 1]
    return expl
