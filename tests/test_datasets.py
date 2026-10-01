import numpy as np
import pandas as pd
import pytest

from fraud.datasets import creditcard, get_dataset, sparkov

# ── contract every dataset must satisfy ─────────────────────────────────────

def test_synthetic_has_canonical_and_raw_columns(spec):
    df = spec.synthetic(500)
    assert {"Time", "Amount", "Class"} <= set(df.columns)
    assert set(spec.raw_cols) <= set(df.columns)
    assert df["Time"].is_monotonic_increasing


def test_features_columns_order_and_no_missing(spec):
    X = spec.add_features(spec.synthetic(500))
    assert list(X.columns) == spec.features
    assert X.isnull().sum().sum() == 0
    assert set(spec.scale_cols) <= set(spec.features)


def test_label_is_never_a_feature(spec):
    assert "Class" not in spec.features


def test_single_row_matches_batch(spec):
    """A transaction scored alone must get the same features as in a batch (no batch statistics)."""
    df = spec.synthetic(500)
    pd.testing.assert_frame_equal(spec.add_features(df).iloc[[10]], spec.add_features(df.iloc[[10]]))


def test_file_round_trip(spec, tmp_path):
    """Writing in the dataset's file format and loading it back gives the same canonical frame."""
    df = spec.synthetic(300)
    path = tmp_path / "tx.csv"
    spec.to_raw(df).to_csv(path, index=False)
    loaded = spec.load(path)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df[loaded.columns].reset_index(drop=True),
                                  check_dtype=False)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("nope")


# ── creditcard ──────────────────────────────────────────────────────────────

def _cc_row(time, amount=1.0):
    return {**dict.fromkeys(creditcard.V_COLS, 0.0), "Time": time, "Amount": amount}


def test_creditcard_hour_is_cyclic():
    X = creditcard.add_features(pd.DataFrame([_cc_row(0), _cc_row(86_400)]))
    cols = ["Hour_sin", "Hour_cos"]
    np.testing.assert_allclose(X.loc[0, cols], X.loc[1, cols], atol=1e-9)


def test_creditcard_negative_amount_is_clipped():
    assert creditcard.add_features(pd.DataFrame([_cc_row(0, amount=-5)]))["Amount_log"].iloc[0] == 0


def test_creditcard_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    creditcard.synthetic(10).drop(columns=["Amount"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Amount"):
        creditcard.load(path)


# ── sparkov ─────────────────────────────────────────────────────────────────

def test_haversine_paris_london():
    assert sparkov.haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(344, abs=2)


def test_sparkov_features_by_hand():
    tx = pd.DataFrame([{
        "Time": (pd.Timestamp("2020-03-07 23:30") - sparkov.EPOCH).total_seconds(),  # a Saturday
        "Amount": 99.0, "card_id": 1, "merchant": "fraud_X", "category": "shopping_net",
        "gender": "F", "dob": "1990-03-07", "lat": 40.0, "long": -75.0, "city_pop": 999,
        "merch_lat": 40.0, "merch_long": -75.0,
    }])
    X = sparkov.add_features(sparkov.enrich(tx)).iloc[0]
    assert X["Amount_log"] == pytest.approx(np.log(100))
    assert X["Day_of_week"] == 5
    assert X["Age"] == pytest.approx(30, abs=0.01)
    assert X["Gender_M"] == 0
    assert X["Distance_km"] == pytest.approx(0)
    assert X["cat_shopping_net"] == 1
    assert X[[c for c in sparkov.FEATURES if c.startswith("cat_")]].sum() == 1
    # a card's first transaction: neutral history features
    assert X["Card_history_log"] == 0
    assert X["First_time_merchant"] == 1
    assert X["Amount_vs_card_mean"] == 1


def test_sparkov_unseen_category_gets_no_one_hot():
    df = sparkov.synthetic(5)
    df["category"] = "crypto"
    assert sparkov.add_features(df)[[c for c in sparkov.FEATURES if c.startswith("cat_")]].sum().sum() == 0


def test_sparkov_loads_a_folder_of_csvs(tmp_path):
    df = sparkov.synthetic(400)
    sparkov.to_raw(df.iloc[:250]).to_csv(tmp_path / "fraudTrain.csv", index=False)
    sparkov.to_raw(df.iloc[250:]).to_csv(tmp_path / "fraudTest.csv", index=False)
    assert len(sparkov.load(tmp_path)) == 400


def test_sparkov_missing_folder_explains_where_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        sparkov.load(tmp_path)


def test_sparkov_never_loads_personal_fields(tmp_path):
    raw = sparkov.to_raw(sparkov.synthetic(50))
    raw["first"], raw["last"], raw["street"] = "Jane", "Doe", "1 Main St"
    raw.to_csv(tmp_path / "tx.csv", index=False)
    assert not {"first", "last", "street"} & set(sparkov.load(tmp_path / "tx.csv").columns)


def test_display_is_readable(spec):
    fields = spec.display(spec.synthetic(300, seed=2).iloc[-1])
    assert fields and all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())
    assert not any(k.startswith("hist_") for k in fields)
    assert not any(str(v).startswith("fraud_") for v in fields.values())
