"""Candidate model pipelines. Every pipeline has a 'prep' step and a 'clf' step."""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TREE_MODELS = (RandomForestClassifier, XGBClassifier, LGBMClassifier)


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
        "LightGBM": lambda p: Pipeline([
            ("prep", make_preprocessor(scale_cols)),
            ("clf", _lgbm(seed, scale_pos_weight=pos_weight, **p)),
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


def _lgbm(seed: int, **params) -> LGBMClassifier:
    # LightGBM's default minimum hessian per leaf (1e-3) lets leaf values explode once predictions
    # saturate under a large scale_pos_weight: on the creditcard data training stopped after 46
    # trees with PR-AUC 0.02. XGBoost's default (1.0) prevents this, so it is used here too.
    params.setdefault("min_child_weight", 1.0)
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
