"""Online card history for real-time scoring.

`history.card_history` computes history columns for a whole table at once (training). A scoring
service sees one transaction at a time, so it keeps a small state per card and must produce the
**same numbers**. `CardHistoryStore` mirrors the batch computation step by step (same running sums,
same additions in the same order), and a test checks exact equality on a replayed dataset.

Per card it keeps: transaction count, running sums (raw, and centred on the card's first amount),
timestamps and running sums of the last 7 days (for the 1 h / 24 h / 7 d windows), merchant and
category counts, and the previous purchase's time and location.

Squares are written x * x, never x ** 2: Python's ** calls the C library's pow(), which is not
guaranteed to round like a multiplication (on Linux it differed in the last bit), and NumPy's
array square is a multiplication. Every other operation used (+, -, *, /, sqrt) is correctly
rounded by IEEE 754, so both paths give identical results on every platform.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from fraud.history import HISTORY_COLS, WINDOWS

_KEEP = max(WINDOWS.values())  # older transactions never fall inside a window again


@dataclass
class _Card:
    n: int = 0
    ref: float = 0.0                # the card's first amount (centring keeps the variance stable)
    inclusive: float = 0.0          # running sums including the latest transaction
    inclusive_c: float = 0.0
    inclusive_sq: float = 0.0
    times: list = field(default_factory=list)     # recent transaction times (last 7 days)
    befores: list = field(default_factory=list)   # sum of earlier amounts, at each recent transaction
    merchants: dict = field(default_factory=lambda: defaultdict(int))
    categories: dict = field(default_factory=lambda: defaultdict(int))
    last_time: float = math.nan
    last_lat: float = math.nan
    last_long: float = math.nan


class OutOfOrderError(ValueError):
    """A card's transaction is older than one already recorded."""


class CardHistoryStore:
    def __init__(self):
        self.cards: dict = {}

    def __len__(self) -> int:
        return len(self.cards)

    def history(self, tx: dict) -> dict:
        """History columns for `tx` (keys: card_id, Time, Amount, merchant, category) from earlier
        transactions only. Does not modify the store."""
        card = self.cards.get(tx["card_id"])
        if card is None or card.n == 0:
            return {**dict.fromkeys(HISTORY_COLS, 0.0), "hist_mean": math.nan, "hist_std": math.nan,
                    "hist_secs_since_prev": math.nan, "hist_prev_lat": math.nan,
                    "hist_prev_long": math.nan}
        t = float(tx["Time"])
        if t < card.last_time:
            raise OutOfOrderError(f"card {tx['card_id']}: transaction at {t} is older than {card.last_time}")
        amount = float(tx["Amount"])
        # Same arithmetic as the batch path: "before" = inclusive running sum - current amount.
        before = (card.inclusive + amount) - amount
        before_c = (card.inclusive_c + (amount - card.ref)) - (amount - card.ref)
        d = amount - card.ref
        before_sq = (card.inclusive_sq + d * d) - d * d
        n = card.n
        out = {}
        for name, seconds in WINDOWS.items():
            start = bisect.bisect_left(card.times, t - seconds)
            out[f"hist_n_{name}"] = len(card.times) - start
            if name == "24h":
                window_before = card.befores[start] if start < len(card.befores) else before
                out["hist_amount_24h"] = before - window_before
        mean_c = before_c / n
        var = (before_sq - n * (mean_c * mean_c)) / (n - 1) if n > 1 else math.nan
        out.update({
            "hist_n": n,
            "hist_mean": mean_c + card.ref,
            "hist_std": math.sqrt(max(var, 0.0)) if n > 1 else math.nan,
            "hist_secs_since_prev": t - card.last_time,
            "hist_merchant_n": card.merchants.get(tx["merchant"], 0),
            "hist_category_n": card.categories.get(tx["category"], 0),
            "hist_prev_lat": card.last_lat,
            "hist_prev_long": card.last_long,
        })
        return {k: out[k] for k in HISTORY_COLS}

    def update(self, tx: dict) -> None:
        """Record `tx` as part of its card's history (call after scoring it)."""
        card = self.cards.get(tx["card_id"])
        if card is None:
            card = self.cards[tx["card_id"]] = _Card(ref=float(tx["Amount"]))
        t, amount = float(tx["Time"]), float(tx["Amount"])
        if card.n and t < card.last_time:
            raise OutOfOrderError(f"card {tx['card_id']}: transaction at {t} is older than {card.last_time}")
        before = (card.inclusive + amount) - amount
        card.inclusive += amount
        card.inclusive_c += amount - card.ref
        card.inclusive_sq += (amount - card.ref) * (amount - card.ref)
        card.times.append(t)
        card.befores.append(before)
        cut = bisect.bisect_left(card.times, t - _KEEP)
        if cut:
            del card.times[:cut], card.befores[:cut]
        card.merchants[tx["merchant"]] += 1
        card.categories[tx["category"]] += 1
        card.n += 1
        card.last_time, card.last_lat, card.last_long = t, float(tx["merch_lat"]), float(tx["merch_long"])

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> CardHistoryStore:
        """Build the state left by a historical table (e.g. the training data) in one pass."""
        store = cls()
        cols = ["card_id", "Time", "Amount", "merchant", "category", "merch_lat", "merch_long"]
        for record in df[cols].sort_values(["card_id", "Time"], kind="stable").itertuples(index=False):
            store.update(record._asdict())
        return store
