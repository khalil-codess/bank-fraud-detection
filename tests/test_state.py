"""The online store must reproduce the batch history used in training, exactly."""

import math

import numpy as np
import pandas as pd
import pytest

from fraud.datasets import sparkov
from fraud.history import HISTORY_COLS, card_history
from fraud.state import CardHistoryStore, OutOfOrderError


def _stream(store, df):
    rows = []
    for tx in df.sort_values("Time", kind="stable").to_dict("records"):
        rows.append(store.history(tx))
        store.update(tx)
    return pd.DataFrame(rows, index=df.sort_values("Time", kind="stable").index)[HISTORY_COLS]


@pytest.fixture(scope="module")
def data():
    df = sparkov.synthetic(4000, seed=11).drop(columns=HISTORY_COLS)
    return df, card_history(df)


def test_streaming_from_empty_matches_batch_exactly(data):
    df, batch = data
    online = _stream(CardHistoryStore(), df)
    pd.testing.assert_frame_equal(online, batch.loc[online.index], check_exact=True, check_dtype=False)


def test_warm_start_then_stream_matches_batch_exactly(data):
    """Production path: state built from history, then new transactions arrive one by one."""
    df, batch = data
    cut = df["Time"].quantile(0.6)
    store = CardHistoryStore.from_frame(df[df["Time"] <= cut])
    later = df[df["Time"] > cut]
    online = _stream(store, later)
    pd.testing.assert_frame_equal(online, batch.loc[online.index], check_exact=True, check_dtype=False)


def test_history_does_not_modify_the_store():
    store = CardHistoryStore()
    tx = {"card_id": 1, "Time": 0, "Amount": 10.0, "merchant": "m", "category": "c",
          "merch_lat": 1.0, "merch_long": 2.0}
    store.update(tx)
    before = store.history({**tx, "Time": 60})
    assert store.history({**tx, "Time": 60}) == before
    assert store.cards[1].n == 1


def test_unknown_card_gets_empty_history():
    h = CardHistoryStore().history({"card_id": 9, "Time": 5, "Amount": 1.0, "merchant": "m", "category": "c"})
    assert h["hist_n"] == 0 and h["hist_n_24h"] == 0 and math.isnan(h["hist_mean"])


def test_out_of_order_transaction_is_rejected():
    store = CardHistoryStore()
    tx = {"card_id": 1, "Time": 100, "Amount": 1.0, "merchant": "m", "category": "c",
          "merch_lat": 0.0, "merch_long": 0.0}
    store.update(tx)
    with pytest.raises(OutOfOrderError):
        store.history({**tx, "Time": 50})
    with pytest.raises(OutOfOrderError):
        store.update({**tx, "Time": 50})


def test_old_transactions_are_pruned_but_totals_kept():
    store = CardHistoryStore()
    for day in range(30):
        store.update({"card_id": 1, "Time": day * 86_400, "Amount": 1.0, "merchant": "m",
                      "category": "c", "merch_lat": 0.0, "merch_long": 0.0})
    assert len(store.cards[1].times) <= 8          # only the last 7 days are kept
    h = store.history({"card_id": 1, "Time": 30 * 86_400, "Amount": 1.0, "merchant": "m", "category": "c"})
    assert h["hist_n"] == 30 and h["hist_n_7d"] == 7 and h["hist_merchant_n"] == 30
    assert h["hist_mean"] == 1.0 and h["hist_std"] == 0.0
    assert np.isclose(h["hist_amount_24h"], 1.0)
