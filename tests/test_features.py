import numpy as np
import pandas as pd

from fraud.features import FEATURES, V_COLS, add_features


def _row(time, amount=1.0):
    return {**dict.fromkeys(V_COLS, 0.0), "Time": time, "Amount": amount}


def test_columns_and_order(synthetic_df):
    assert list(add_features(synthetic_df).columns) == FEATURES


def test_no_missing_values(synthetic_df):
    assert add_features(synthetic_df).isnull().sum().sum() == 0


def test_label_and_raw_time_are_not_features(synthetic_df):
    X = add_features(synthetic_df)
    assert "Class" not in X.columns
    assert "Time" not in X.columns


def test_single_row_matches_batch(synthetic_df):
    """A transaction scored alone must get the same features as in a batch (no batch statistics)."""
    pd.testing.assert_frame_equal(add_features(synthetic_df).iloc[[10]],
                                  add_features(synthetic_df.iloc[[10]]))


def test_hour_is_cyclic():
    X = add_features(pd.DataFrame([_row(0), _row(86_400)]))
    np.testing.assert_allclose(X.loc[0, ["Hour_sin", "Hour_cos"]],
                               X.loc[1, ["Hour_sin", "Hour_cos"]], atol=1e-9)


def test_negative_amount_is_clipped():
    assert add_features(pd.DataFrame([_row(0, amount=-5)]))["Amount_log"].iloc[0] == 0
