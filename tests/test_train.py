"""End-to-end: train on a small synthetic CSV and check the artifacts the dashboard needs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fraud.config import load_config
from fraud.inference import load_artifacts, score_transactions
from fraud.synthetic import make_synthetic
from fraud.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    data = tmp / "tx.csv"
    make_synthetic(n=4000, seed=1).to_csv(data, index=False)
    cfg = load_config("config.yaml").with_overrides(
        data_path=data, artifacts_dir=tmp / "artifacts", outputs_dir=tmp / "outputs", shap=False)
    return cfg, run(cfg)


def test_artifacts_are_written(trained):
    cfg, _ = trained
    for name in ("fraud_model.joblib", "metrics.json", "demo_transactions.csv"):
        assert (cfg.artifacts_dir / name).exists(), name
    for name in ("eda_distribution.png", "pr_curves.png", "confusion_matrix.png"):
        assert (cfg.outputs_dir / name).exists(), name


def test_metrics_are_consistent(trained):
    cfg, metrics = trained
    on_disk = json.loads((cfg.artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["selected_model"] == metrics["selected_model"]
    test = on_disk["models"][on_disk["selected_model"]]["test"]
    assert test["tp"] + test["fn"] == on_disk["data"]["test"]["frauds"]


def test_saved_model_flags_frauds(trained):
    cfg, _ = trained
    model, metrics = load_artifacts(cfg.artifacts_dir)
    demo = pd.read_csv(cfg.artifacts_dir / "demo_transactions.csv")
    flagged = score_transactions(model, metrics["threshold"], demo)["is_fraud"]
    assert flagged[demo["Class"] == 1].mean() > flagged[demo["Class"] == 0].mean() + 0.5


@pytest.mark.skipif(not Path("artifacts/fraud_model.joblib").exists(),
                    reason="real model not trained (python -m fraud.train)")
def test_real_model_matches_current_features():
    _, metrics = load_artifacts("artifacts")
    from fraud.features import FEATURES
    assert metrics["features"] == FEATURES
