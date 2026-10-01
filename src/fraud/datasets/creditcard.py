"""Kaggle "Credit Card Fraud Detection" (ULB, 2013): 2 days, 28 anonymised PCA components.

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.datasets import DatasetSpec

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = V_COLS + ["Time", "Amount"]
FEATURES = V_COLS + ["Amount_log", "Hour_sin", "Hour_cos"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLS + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[RAW_COLS + ["Class"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nothing is learned here, so training and serving compute identical values."""
    out = df[V_COLS].copy()
    out["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    # Time = seconds since the first transaction in the file -> position in the 24 h cycle
    hour = (df["Time"] // 3600) % 24
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURES]


REASON_GROUPS = {**{v: v for v in V_COLS}, "Amount_log": "amount",
                 "Hour_sin": "time_of_day", "Hour_cos": "time_of_day"}


def describe(group: str, raw: pd.Series, feats: pd.Series) -> str:
    if group == "amount":
        return f"Amount {raw['Amount']:,.2f}"
    if group == "time_of_day":
        return f"Hour {int(raw['Time'] // 3600 % 24)} of the daily cycle (clock time is not in the data)"
    return f"Anonymised component {group} = {raw[group]:.2f}"


def display(raw: pd.Series) -> dict:
    return {"Time": f"{raw['Time'] / 3600:.1f} h after the first transaction",
            "Amount": f"{raw['Amount']:,.2f}",
            "V1–V28": "28 anonymised PCA components (not shown)"}


def synthetic(n: int = 3000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Random transactions in this schema; frauds have shifted V1/V2 so a model can learn them."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = rng.exponential(80, n).round(2)
    df["Class"] = (rng.random(n) < fraud_rate).astype(int)
    df.loc[df["Class"] == 1, ["V1", "V2"]] += 4
    return df


SPEC = DatasetSpec(
    name="creditcard",
    description="Kaggle ULB credit-card transactions (2013, 2 days); V1–V28 are anonymised PCA components.",
    raw_cols=RAW_COLS,
    features=FEATURES,
    scale_cols=["Amount_log"],
    load=load,
    add_features=add_features,
    synthetic=synthetic,
    reason_groups=REASON_GROUPS,
    describe=describe,
    display=display,
)
