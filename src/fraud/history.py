"""Point-in-time card history.

For every transaction, `card_history` summarises what had happened on the same card **before**
it: counts and amounts in recent windows, the card's usual amount, time since its previous
transaction, earlier visits to the merchant and category, and where the previous purchase was.

Two rules make these features safe to train on and to serve:
  - only earlier transactions of the card are used (never the current or later ones), so a row's
    history is identical whether or not future data exists;
  - fraud labels are never used: in production they arrive weeks after the transaction.

Rows are ordered by (card, Time); transactions of one card with the same timestamp are ordered as
they appear in the input, and each sees only those before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400}
HISTORY_COLS = (
    [f"hist_n_{w}" for w in WINDOWS]
    + ["hist_amount_24h", "hist_n", "hist_mean", "hist_std", "hist_secs_since_prev",
       "hist_merchant_n", "hist_category_n", "hist_prev_lat", "hist_prev_long"]
)


def card_history(df: pd.DataFrame) -> pd.DataFrame:
    """History columns for every row of `df` (columns: card_id, Time, Amount, merchant,
    category, merch_lat, merch_long), aligned on `df.index`."""
    s = df[["card_id", "Time", "Amount", "merchant", "category", "merch_lat", "merch_long"]]
    s = s.sort_values(["card_id", "Time"], kind="stable")
    n = len(s)
    if n == 0:
        return pd.DataFrame(columns=HISTORY_COLS, index=df.index, dtype=float)

    card = pd.factorize(s["card_id"])[0]                  # increasing, because rows are sorted by card
    t = s["Time"].to_numpy(dtype=float)
    amount = s["Amount"].to_numpy(dtype=float)
    pos = np.arange(n)
    is_start = np.r_[True, card[1:] != card[:-1]]
    group_start = np.maximum.accumulate(np.where(is_start, pos, 0))
    prior_n = pos - group_start                            # earlier transactions of this card

    # One sorted key across cards, spaced so a window can never reach into the previous card.
    span = t.max() - t.min() + max(WINDOWS.values()) + 1
    key = card * span + (t - t.min())

    # Running sums restart for every card and are centred on the card's first amount (known to all
    # its later rows). Sums never mix cards, so floating-point error cannot depend on other cards'
    # later rows, and centring keeps the variance numerically stable.
    ref = amount[group_start]
    shifted = amount - ref
    before = _exclusive_cumsum(amount, card)            # sum of the card's earlier amounts
    before_c = _exclusive_cumsum(shifted, card)
    before_sq = _exclusive_cumsum(shifted**2, card)

    out = {}
    for name, seconds in WINDOWS.items():
        window_start = np.searchsorted(key, key - seconds, side="left")
        out[f"hist_n_{name}"] = pos - window_start
        if name == "24h":
            out["hist_amount_24h"] = before - before[window_start]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = np.where(prior_n > 0, before_c / prior_n, np.nan)
        var = np.where(prior_n > 1, (before_sq - prior_n * mean_c**2) / (prior_n - 1), np.nan)
    out["hist_n"] = prior_n
    out["hist_mean"] = mean_c + ref
    out["hist_std"] = np.sqrt(np.clip(var, 0, None))

    has_prev = ~is_start
    prev = np.maximum(pos - 1, 0)
    out["hist_secs_since_prev"] = np.where(has_prev, t - t[prev], np.nan)
    out["hist_prev_lat"] = np.where(has_prev, s["merch_lat"].to_numpy(dtype=float)[prev], np.nan)
    out["hist_prev_long"] = np.where(has_prev, s["merch_long"].to_numpy(dtype=float)[prev], np.nan)
    out["hist_merchant_n"] = s.groupby(["card_id", "merchant"], sort=False).cumcount().to_numpy()
    out["hist_category_n"] = s.groupby(["card_id", "category"], sort=False).cumcount().to_numpy()

    return pd.DataFrame(out, index=s.index)[HISTORY_COLS].reindex(df.index)


def _exclusive_cumsum(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Per-group running sum of the *earlier* values (0 for a group's first row).

    Plain sequential addition per group (np.cumsum), so the online store (state.py), which adds
    one transaction at a time, reproduces it bit for bit. Rows must be sorted by group.
    """
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    inclusive = np.concatenate([np.cumsum(chunk) for chunk in np.split(values, starts[1:])])
    return inclusive - values
