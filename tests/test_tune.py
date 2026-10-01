import json

import pytest
from conftest import config_for

pytest.importorskip("optuna")
from fraud.datasets import get_dataset  # noqa: E402
from fraud.tune import tune  # noqa: E402


@pytest.mark.parametrize("model", ["LightGBM", "XGBoost"])
def test_tune_writes_best_params_and_never_uses_validation_or_test(tmp_path, model):
    spec = get_dataset("sparkov")
    data = tmp_path / "tx.csv"
    spec.to_raw(spec.synthetic(n=4000, seed=2)).to_csv(data, index=False)
    cfg = config_for("sparkov").with_overrides(data_path=data, outputs_dir=tmp_path / "out")
    result = tune(cfg, model, trials=3, negatives=0.5)

    on_disk = json.loads(next((tmp_path / "out").glob("tuning_*.json")).read_text(encoding="utf-8"))
    assert on_disk["best_pr_auc"] == result["best_pr_auc"] >= max(on_disk["history"]) - 1e-12
    assert on_disk["trials"] == 3
    # fit + holdout come from the training window only (70% of the data with default 15/15 splits)
    assert on_disk["fit_rows"] < 4000 * 0.7 * 0.8 + 1
    assert on_disk["holdout_rows"] == 4000 * 0.7 - int(4000 * 0.7 * 0.8)
