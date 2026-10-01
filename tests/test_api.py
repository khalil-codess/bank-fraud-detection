"""The API, end to end, on a model trained on synthetic Sparkov data."""

import numpy as np
import pandas as pd
import pytest
from conftest import config_for

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fraud.api import create_app  # noqa: E402
from fraud.data import load_transactions  # noqa: E402
from fraud.datasets import sparkov  # noqa: E402
from fraud.history import HISTORY_COLS  # noqa: E402
from fraud.inference import load_artifacts, score_transactions  # noqa: E402
from fraud.train import run  # noqa: E402


def _payload(row) -> dict:
    return {
        "transaction_id": str(row.name), "card_id": int(row["card_id"]),
        "timestamp": (sparkov.EPOCH + pd.to_timedelta(row["Time"], unit="s")).isoformat(),
        "amount": float(row["Amount"]), "merchant": row["merchant"], "category": row["category"],
        "gender": row["gender"], "dob": row["dob"], "lat": float(row["lat"]), "long": float(row["long"]),
        "city_pop": int(row["city_pop"]), "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
    }


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Train on synthetic data; the API warms its history from the older 85% of the file."""
    tmp = tmp_path_factory.mktemp("api")
    df = sparkov.synthetic(n=3000, seed=7).drop(columns=HISTORY_COLS)
    cut = df["Time"].quantile(0.85)
    sparkov.to_raw(df).to_csv(tmp / "all.csv", index=False)
    (tmp / "history").mkdir()
    sparkov.to_raw(df[df["Time"] <= cut]).to_csv(tmp / "history" / "tx.csv", index=False)
    cfg = config_for("sparkov").with_overrides(
        data_path=tmp / "all.csv", artifacts_dir=tmp / "art", outputs_dir=tmp / "out", shap=False, cv_folds=0)
    run(cfg)
    return tmp, df, cut


@pytest.fixture(scope="module")
def client(setup):
    tmp, _, _ = setup
    with TestClient(create_app(artifacts=tmp / "art", history=tmp / "history")) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["dataset"] == "sparkov" and body["cards_in_history"] > 0


def test_streamed_decisions_match_offline_scoring(setup):
    """No training/serving skew: replaying the last 15% through the API gives the same scores
    as scoring the whole file offline with batch history."""
    tmp, df, cut = setup
    model, metrics, spec = load_artifacts(tmp / "art")
    offline_all = load_transactions(spec, tmp / "all.csv")
    later = offline_all[offline_all["Time"] > cut].sort_values("Time", kind="stable")
    offline = score_transactions(model, metrics["policy"], later, spec)

    with TestClient(create_app(artifacts=tmp / "art", history=tmp / "history")) as c:
        online = [c.post("/score", json=_payload(row)).json() for _, row in later.iterrows()]
    np.testing.assert_allclose([r["fraud_probability"] for r in online], offline["fraud_proba"], rtol=1e-6)
    assert [r["decision"] == "review" for r in online] == offline["is_fraud"].tolist()
    assert [r["card_transactions_before"] for r in online] == later["hist_n"].astype(int).tolist()


def test_response_shape_and_reasons(client, setup):
    _, df, cut = setup
    row = df[(df["Time"] > cut) & (df["Class"] == 1)].iloc[0]
    body = client.post("/score?record=false", json=_payload(row)).json()
    assert body["decision"] in {"review", "approve"}
    assert 0 <= body["fraud_probability"] <= 1
    assert body["expected_loss"] == pytest.approx(body["fraud_probability"] * row["Amount"])
    assert 1 <= len(body["reasons"]) <= 3 and all(r["impact"] > 0 for r in body["reasons"])
    assert body["recorded"] is False and body["latency_ms"] > 0


def test_record_false_leaves_history_untouched(client, setup):
    _, df, cut = setup
    row = df[df["Time"] > cut].iloc[-1]
    first = client.post("/score?record=false", json=_payload(row)).json()
    second = client.post("/score?record=false", json=_payload(row)).json()
    assert first["card_transactions_before"] == second["card_transactions_before"]


def test_validation_errors(client, setup):
    _, df, _ = setup
    bad = _payload(df.iloc[0]) | {"amount": -5}
    assert client.post("/score", json=bad).status_code == 422
    bad = _payload(df.iloc[0]) | {"gender": "X"}
    assert client.post("/score", json=bad).status_code == 422


def test_out_of_order_transaction_is_rejected(client, setup):
    _, df, cut = setup
    old = _payload(df[df["Time"] <= cut].iloc[0])  # a card's very first transaction, long ago
    card_has_later_history = df[(df["card_id"] == old["card_id"]) & (df["Time"] <= cut)].shape[0] > 1
    if card_has_later_history:
        assert client.post("/score", json=old).status_code == 409
