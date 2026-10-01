"""End-to-end: train on a small synthetic file for each dataset and check the dashboard's artifacts."""

import json

import pandas as pd
import pytest
from conftest import config_for

from fraud.calibration import CalibratedModel
from fraud.datasets import DATASETS, get_dataset
from fraud.inference import load_artifacts, score_transactions
from fraud.train import run


@pytest.fixture(scope="module", params=DATASETS)
def trained(request, tmp_path_factory):
    spec = get_dataset(request.param)
    tmp = tmp_path_factory.mktemp(spec.name)
    data = tmp / "tx.csv"
    spec.to_raw(spec.synthetic(n=4000, seed=1)).to_csv(data, index=False)
    cfg = config_for(spec.name).with_overrides(
        data_path=data, artifacts_dir=tmp / "artifacts", outputs_dir=tmp / "outputs", shap=False)
    return cfg, run(cfg)


def test_artifacts_are_written(trained):
    cfg, _ = trained
    for name in ("fraud_model.joblib", "metrics.json", "demo_transactions.csv"):
        assert (cfg.artifacts_dir / name).exists(), name
    for name in ("eda_distribution.png", "pr_curves.png", "confusion_matrix.png",
                 "savings_curve.png", "cv_results.png"):
        assert (cfg.outputs_dir / name).exists(), name


def test_metrics_are_consistent(trained):
    cfg, metrics = trained
    on_disk = json.loads((cfg.artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["dataset"] == cfg.dataset
    assert on_disk["selected_model"] == metrics["selected_model"]
    test = on_disk["models"][on_disk["selected_model"]]["test"]
    assert test["tp"] + test["fn"] == on_disk["data"]["test"]["frauds"]


def test_saved_model_flags_frauds(trained):
    cfg, _ = trained
    model, metrics, spec = load_artifacts(cfg.artifacts_dir)
    assert spec.name == cfg.dataset
    demo = pd.read_csv(cfg.artifacts_dir / "demo_transactions.csv")
    flagged = score_transactions(model, metrics["policy"], demo, spec)["is_fraud"]
    assert flagged[demo["Class"] == 1].mean() > flagged[demo["Class"] == 0].mean() + 0.5


def test_saved_model_is_calibrated_and_keeps_the_ranking(trained):
    """Calibration is monotone: the final model ranks exactly like the selected raw model."""
    cfg, metrics = trained
    model, _, _ = load_artifacts(cfg.artifacts_dir)
    assert isinstance(model, CalibratedModel) and model.slope > 0
    raw = metrics["models"][metrics["selected_model"]]["test"]
    assert metrics["final"]["pr_auc"] == pytest.approx(raw["pr_auc"], abs=1e-9)


def test_decision_rule_chosen_on_validation(trained):
    _, metrics = trained
    decision = metrics["decision"]
    assert set(decision) == {"threshold", "expected_value"}  # both rules evaluated (cost method)
    chosen = metrics["policy"]["rule"]
    assert decision[chosen]["val"]["savings"] == max(d["val"]["savings"] for d in decision.values())
    assert metrics["final"] == decision[chosen]["test"]
    final = metrics["final"]
    assert final["tp"] + final["fn"] == metrics["data"]["test"]["frauds"]
