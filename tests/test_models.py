import numpy as np
import pytest

from fraud.data import time_split
from fraud.evaluate import evaluate
from fraud.inference import explain, score_transactions
from fraud.models import build_models


@pytest.fixture(scope="module")
def fitted(module_spec, module_config):
    """Every configured model, fitted on synthetic data of each dataset."""
    spec = module_spec
    train, _, test = time_split(spec.synthetic(4000, seed=0))
    models = build_models(train["Class"], module_config.models, spec.scale_cols, seed=0)
    for model in models.values():
        model.fit(spec.add_features(train), train["Class"])
    return spec, models, train, test


def test_every_configured_model_beats_chance(fitted):
    spec, models, _, test = fitted
    for name, model in models.items():
        p = model.predict_proba(spec.add_features(test))[:, 1]
        assert np.all((p >= 0) & (p <= 1)), name
        assert evaluate(test["Class"], p, 0.5)["roc_auc"] > 0.9, name


def test_unknown_model_name_is_rejected(fitted):
    spec, _, train, _ = fitted
    with pytest.raises(ValueError, match="Unknown models"):
        build_models(train["Class"], {"Magic Model": {}}, spec.scale_cols)


def test_scaler_is_fitted_on_training_data_only(fitted):
    spec, models, train, _ = fitted
    scaler = models["XGBoost"].named_steps["prep"].named_transformers_["scale"]
    expected = spec.add_features(train)[spec.scale_cols].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)


def test_smote_does_not_resample_at_prediction(fitted):
    spec, models, _, test = fitted
    assert models["XGBoost + SMOTE"].predict_proba(spec.add_features(test.iloc[:7])).shape == (7, 2)


def test_score_transactions_from_raw_columns(fitted):
    spec, models, _, test = fitted
    out = score_transactions(models["Random Forest"], 0.5, test[spec.raw_cols], spec)
    assert list(out.columns) == ["fraud_proba", "is_fraud"]
    assert len(out) == len(test)


def test_extreme_amounts_do_not_crash(fitted):
    spec, models, _, test = fitted
    extreme = test.iloc[:3].copy()
    extreme["Amount"] = 1e7
    assert len(score_transactions(models["XGBoost"], 0.5, extreme, spec)) == 3


@pytest.mark.parametrize("name", ["Random Forest", "XGBoost"])
def test_shap_explanation_shape(fitted, name):
    spec, models, _, test = fitted
    expl = explain(models[name], test.iloc[:4], spec)
    assert expl.values.shape == (4, len(spec.features))
