import pandas as pd
import pytest

from fraud.data import load_transactions, time_split
from fraud.synthetic import make_synthetic


def test_split_is_chronological_without_overlap():
    train, val, test = time_split(make_synthetic().sample(frac=1, random_state=1))
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_split_covers_all_rows_with_expected_sizes():
    train, val, test = time_split(make_synthetic(n=3000), val_frac=0.15, test_frac=0.15)
    assert len(train) + len(val) + len(test) == 3000
    assert len(val) == len(test) == 450


def test_load_removes_duplicates(tmp_path):
    df = make_synthetic(n=100)
    path = tmp_path / "tx.csv"
    pd.concat([df, df.iloc[:5]]).to_csv(path, index=False)
    loaded = load_transactions(path)
    assert len(loaded) == 100
    assert loaded.attrs["rows_raw"] == 105


def test_load_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    make_synthetic(n=10).drop(columns=["Amount"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Amount"):
        load_transactions(path)
