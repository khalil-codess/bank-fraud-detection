"""Candidate model pipelines. Every pipeline has a 'prep' step and a 'clf' step."""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TREE_MODELS = (RandomForestClassifier, XGBClassifier)


def make_preprocessor(scale_cols: list[str]) -> ColumnTransformer:
    """Standardise numeric features; the scaler is fitted on the training data only."""
    return ColumnTransformer(
        [("scale", StandardScaler(), list(scale_cols))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_models(y_train, params: dict, scale_cols: list[str], seed: int = 42) -> dict:
    """Build the pipelines named in `params` (the `models` section of the config)."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    builders = {
        "Logistic Regression": lambda p: Pipeline([
            ("prep", make_preprocessor(scale_cols)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=seed, **p)),
        ]),
        "Random Forest": lambda p: Pipeline([
            ("prep", make_preprocessor(scale_cols)),
            ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                           random_state=seed, n_jobs=-1, **p)),
        ]),
        "XGBoost": lambda p: Pipeline([
            ("prep", make_preprocessor(scale_cols)),
            ("clf", _xgb(seed, scale_pos_weight=pos_weight, **p)),
        ]),
        # Ablation: SMOTE resamples the training data only (imblearn skips it at predict time)
        "XGBoost + SMOTE": lambda p: ImbPipeline([
            ("prep", make_preprocessor(scale_cols)),
            ("smote", SMOTE(sampling_strategy=p.pop("smote_ratio", 0.1), random_state=seed)),
            ("clf", _xgb(seed, **p)),
        ]),
    }
    unknown = set(params) - set(builders)
    if unknown:
        raise ValueError(f"Unknown models in config: {sorted(unknown)}")
    return {name: builders[name](dict(p or {})) for name, p in params.items()}


def _xgb(seed: int, **params) -> XGBClassifier:
    return XGBClassifier(eval_metric="aucpr", random_state=seed, n_jobs=-1, verbosity=0, **params)
