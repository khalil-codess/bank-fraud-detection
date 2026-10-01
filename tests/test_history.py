"""Card history must be point-in-time correct: no future rows, no labels."""

import numpy as np
import pandas as pd
import pytest

from fraud.datasets import sparkov
from fraud.history import HISTORY_COLS, card_history

HOUR = 3_600


def _tx(card, time, amount, merchant="m1", category="grocery_pos", lat=40.0, long=-75.0):
    return {"card_id": card, "Time": time, "Amount": amount, "merchant": merchant,
            "category": category, "merch_lat": lat, "merch_long": long}


@pytest.fixture
def small():
    """Card A: four purchases; card B: one purchase in between. Rows deliberately unsorted."""
    rows = [
        _tx("A", 10 * HOUR, 40.0, merchant="m2", category="travel", lat=41.0),
        _tx("A", 0, 10.0),
        _tx("B", 5 * HOUR, 999.0),
        _tx("A", 30 * 60, 20.0),
        _tx("A", 10 * HOUR + 60, 30.0),
    ]
    return pd.DataFrame(rows, index=[100, 101, 102, 103, 104])


def test_values_by_hand(small):
    h = card_history(small)
    first, second, third, fourth = h.loc[101], h.loc[103], h.loc[100], h.loc[104]

    assert first["hist_n"] == 0 and np.isnan(first["hist_mean"]) and np.isnan(first["hist_secs_since_prev"])
    assert second["hist_n"] == 1 and second["hist_n_1h"] == 1 and second["hist_mean"] == 10
    assert second["hist_secs_since_prev"] == 30 * 60
    # 10 h after the first purchase: both earlier ones fall in the 24 h window, none in the last hour
    assert (third["hist_n_1h"], third["hist_n_24h"], third["hist_amount_24h"]) == (0, 2, 30.0)
    assert third["hist_mean"] == 15 and third["hist_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert third["hist_merchant_n"] == 0 and third["hist_category_n"] == 0   # first time at m2 / travel
    # one minute later, back at m1: previous purchase was at m2 (lat 41)
    assert fourth["hist_n_1h"] == 1 and fourth["hist_merchant_n"] == 2 and fourth["hist_prev_lat"] == 41.0


def test_cards_never_mix(small):
    h = card_history(small)
    assert h.loc[102, "hist_n"] == 0            # card B's only purchase sees nothing of card A
    assert h.loc[100, "hist_n_24h"] == 2        # card A never counts card B's 999 purchase


def test_aligned_on_input_index(small):
    h = card_history(small)
    assert list(h.index) == list(small.index)
    assert list(h.columns) == HISTORY_COLS


def test_no_look_ahead_on_real_scale_data():
    """A row's history is identical whether or not later transactions exist."""
    df = sparkov.synthetic(3000, seed=3)
    full = card_history(df)
    for cut in np.quantile(df["Time"], [0.25, 0.5, 0.75]):
        past = df[df["Time"] <= cut]
        pd.testing.assert_frame_equal(card_history(past), full.loc[past.index], check_exact=True)


def test_changing_the_future_changes_nothing():
    df = sparkov.synthetic(2000, seed=4)
    cut = df["Time"].median()
    altered = df.copy()
    future = altered["Time"] > cut
    altered.loc[future, "Amount"] *= 100
    altered.loc[future, "merchant"] = "somewhere_else"
    past = df["Time"] <= cut
    pd.testing.assert_frame_equal(card_history(altered)[past], card_history(df)[past], check_exact=True)


def test_labels_are_never_used():
    df = sparkov.synthetic(2000, seed=5)
    flipped = df.assign(Class=1 - df["Class"])
    pd.testing.assert_frame_equal(sparkov.enrich(flipped)[HISTORY_COLS], sparkov.enrich(df)[HISTORY_COLS])


def test_amount_what_if_updates_comparison_features():
    """History describes only the past, so changing the current amount still moves these features."""
    df = sparkov.synthetic(2000, seed=6)
    row = df[df["hist_n"] > 5].iloc[[0]]
    base = sparkov.add_features(row).iloc[0]
    bigger = sparkov.add_features(row.assign(Amount=row["Amount"] * 10)).iloc[0]
    assert bigger["Amount_vs_card_mean"] == pytest.approx(base["Amount_vs_card_mean"] * 10)
    assert bigger["Amount_zscore_card"] > base["Amount_zscore_card"]
