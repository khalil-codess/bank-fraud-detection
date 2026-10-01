"""Sparkov simulated credit-card transactions (2019-2020, ~1.85M rows, ~1,000 cards, 800 merchants).

https://www.kaggle.com/datasets/kartik2112/fraud-detection
Put fraudTrain.csv and fraudTest.csv in one folder (default data/raw/sparkov/); both are loaded and
re-split chronologically. The data is synthetic (generated with the Sparkov tool): names and
addresses are fake, and they are dropped on load anyway.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.datasets import DatasetSpec

EPOCH = pd.Timestamp("2019-01-01")
CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]
# Raw CSV column -> canonical name. Personal fields (name, street, job, ...) are never loaded.
COLUMNS = {
    "trans_date_trans_time": "timestamp", "cc_num": "card_id", "merchant": "merchant",
    "category": "category", "amt": "Amount", "gender": "gender", "dob": "dob", "lat": "lat",
    "long": "long", "city_pop": "city_pop", "merch_lat": "merch_lat", "merch_long": "merch_long",
    "is_fraud": "Class",
}
RAW_COLS = ["Time", "Amount", "card_id", "merchant", "category", "gender", "dob",
            "lat", "long", "city_pop", "merch_lat", "merch_long"]
BASE_FEATURES = ["Amount_log", "Hour_sin", "Hour_cos", "Day_of_week", "Age", "Gender_M",
                 "City_pop_log", "Distance_km"]
FEATURES = BASE_FEATURES + [f"cat_{c}" for c in CATEGORIES]


def load(path: Path) -> pd.DataFrame:
    """Load every CSV in `path` (a folder) or the single file `path`."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV found in {path}. Download fraudTrain.csv and fraudTest.csv "
                                "from https://www.kaggle.com/datasets/kartik2112/fraud-detection")
    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{f} is missing columns: {missing}")
        frames.append(pd.read_csv(f, usecols=list(COLUMNS)).rename(columns=COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df["Time"] = (pd.to_datetime(df.pop("timestamp")) - EPOCH).dt.total_seconds()
    return df[RAW_COLS + ["Class"]]


def to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical frame -> Kaggle column names and timestamp format (inverse of `load`)."""
    out = df.copy()
    out["timestamp"] = timestamps(out.pop("Time")).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.rename(columns={v: k for k, v in COLUMNS.items()})


def timestamps(df: pd.DataFrame | pd.Series) -> pd.Series:
    seconds = df["Time"] if isinstance(df, pd.DataFrame) else df
    return EPOCH + pd.to_timedelta(seconds, unit="s")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features (no history yet). Deterministic: identical in training and serving."""
    ts = timestamps(df)
    out = pd.DataFrame(index=df.index)
    out["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    hour = ts.dt.hour + ts.dt.minute / 60
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["Day_of_week"] = ts.dt.dayofweek
    out["Age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["Gender_M"] = (df["gender"] == "M").astype(int)
    out["City_pop_log"] = np.log1p(df["city_pop"].clip(lower=0))
    out["Distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    for c in CATEGORIES:  # fixed list: an unseen category simply gets all zeros
        out[f"cat_{c}"] = (df["category"] == c).astype(int)
    return out[FEATURES]


def synthetic(n: int = 3000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Transactions in the Sparkov schema. Frauds are larger, at night and in online categories."""
    rng = np.random.default_rng(seed)
    n_cards = max(n // 40, 5)
    cards = pd.DataFrame({
        "card_id": rng.integers(10**15, 10**16, n_cards),
        "gender": rng.choice(["F", "M"], n_cards),
        "dob": pd.to_datetime(rng.integers(-10_000, 10_000, n_cards), unit="D").strftime("%Y-%m-%d"),
        "lat": rng.uniform(30, 45, n_cards), "long": rng.uniform(-120, -75, n_cards),
        "city_pop": rng.integers(100, 2_000_000, n_cards),
    })
    df = cards.iloc[rng.integers(0, n_cards, n)].reset_index(drop=True)
    df["Time"] = np.sort(rng.uniform(0, 90 * 86_400, n)).round()
    df["Class"] = (rng.random(n) < fraud_rate).astype(int)
    fraud = df["Class"] == 1
    df["category"] = rng.choice(CATEGORIES, n)
    df.loc[fraud, "category"] = rng.choice(["shopping_net", "misc_net", "grocery_pos"], fraud.sum())
    df["Amount"] = rng.exponential(60, n).round(2)
    df.loc[fraud, "Amount"] = rng.uniform(300, 1200, fraud.sum()).round(2)
    night = rng.uniform(22, 27, fraud.sum()) % 24 * 3600  # frauds between 22:00 and 03:00
    df.loc[fraud, "Time"] = (df.loc[fraud, "Time"] // 86_400) * 86_400 + night.round()
    df["merchant"] = "fraud_" + df["category"] + "_" + rng.integers(0, 20, n).astype(str)
    df["merch_lat"] = df["lat"] + rng.uniform(-1, 1, n)
    df["merch_long"] = df["long"] + rng.uniform(-1, 1, n)
    return df.sort_values("Time", kind="stable").reset_index(drop=True)[RAW_COLS + ["Class"]]


SPEC = DatasetSpec(
    name="sparkov",
    description="Sparkov simulated card transactions (2019–2020) with card, merchant, category and location.",
    raw_cols=RAW_COLS,
    features=FEATURES,
    scale_cols=["Amount_log", "Day_of_week", "Age", "City_pop_log", "Distance_km"],
    load=load,
    add_features=add_features,
    synthetic=synthetic,
    to_raw=to_raw,
)
