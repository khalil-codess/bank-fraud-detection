import numpy as np
import pytest

from fraud.evaluate import evaluate
from fraud.features import V_COLS, add_features
from fraud.inference import explain, score_transactions
from fraud.models import build_models


@pytest.fixture(scope="module")
def fitted(splits, model_params):
    train, _, _ = splits
    models = build_models(train["Class"], model_params, seed=0)
    for model in models.values():
        model.fit(add_features(train), train["Class"])
    return models


def test_every_configured_model_beats_chance(fitted, splits):
    _, _, test = splits
    for name, model in fitted.items():
        p = model.predict_proba(add_features(test))[:, 1]
        assert np.all((p >= 0) & (p <= 1)), name
        assert evaluate(test["Class"], p, 0.5)["roc_auc"] > 0.9, name


def test_unknown_model_name_is_rejected(splits):
    with pytest.raises(ValueError, match="Unknown models"):
        build_models(splits[0]["Class"], {"Magic Model": {}})


def test_scaler_is_fitted_on_training_data_only(fitted, splits):
    train, _, _ = splits
    scaler = fitted["XGBoost"].named_steps["prep"].named_transformers_["amount"]
    assert scaler.mean_[0] == pytest.approx(add_features(train)["Amount_log"].mean())


def test_smote_does_not_resample_at_prediction(fitted, splits):
    rows = add_features(splits[2].iloc[:7])
    assert fitted["XGBoost + SMOTE"].predict_proba(rows).shape == (7, 2)


def test_score_transactions_from_raw_columns(fitted, splits):
    test = splits[2]
    out = score_transactions(fitted["Random Forest"], 0.5, test)
    assert list(out.columns) == ["fraud_proba", "is_fraud"]
    assert len(out) == len(test)


def test_extreme_values_do_not_crash(fitted, splits):
    extreme = splits[2].iloc[:3].copy()
    extreme[V_COLS] = 999.0
    extreme["Amount"] = 1e7
    assert len(score_transactions(fitted["XGBoost"], 0.5, extreme)) == 3


@pytest.mark.parametrize("name", ["Random Forest", "XGBoost"])
def test_shap_explanation_shape(fitted, splits, name):
    expl = explain(fitted[name], splits[2].iloc[:4])
    assert expl.values.shape == (4, len(add_features(splits[2]).columns))
