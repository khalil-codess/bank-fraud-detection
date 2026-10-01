import pytest

from fraud.config import Config
from fraud.synthetic import make_synthetic
from fraud.validation import cross_validate, rolling_folds


def test_folds_are_chronological_and_never_look_ahead():
    df = make_synthetic(n=2000)
    folds = list(rolling_folds(df, n_folds=4, fold_frac=0.1))
    assert len(folds) == 4
    for fit, calib, test in folds:
        assert fit["Time"].max() <= calib["Time"].min()
        assert calib["Time"].max() <= test["Time"].min()
        assert len(test) == 200
    # expanding window: each fold trains on more history than the previous one
    sizes = [len(fit) + len(calib) for fit, _, _ in folds]
    assert sizes == sorted(sizes)
    # test blocks are consecutive and do not overlap
    for (_, _, a), (_, _, b) in zip(folds, folds[1:]):
        assert a["Time"].max() <= b["Time"].min()


def test_too_many_folds_is_rejected():
    with pytest.raises(ValueError):
        list(rolling_folds(make_synthetic(n=100), n_folds=10, fold_frac=0.1))


def test_cross_validate_summaries(model_params):
    cfg = Config(cv_folds=2, cv_fold_frac=0.2, models={"XGBoost": model_params["XGBoost"]})
    cv = cross_validate(make_synthetic(n=3000), cfg)
    assert len(cv["XGBoost"]["folds"]) == 2
    for metric in ("pr_auc", "savings_rate", "recall"):
        assert set(cv["XGBoost"][metric]) == {"mean", "std"}
    assert cv["XGBoost"]["pr_auc"]["mean"] > 0.8


def test_invalid_threshold_method_is_rejected():
    with pytest.raises(ValueError):
        Config(threshold_method="magic")
