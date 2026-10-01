"""Loading and chronological splitting of transaction data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud.features import RAW_COLS

log = logging.getLogger(__name__)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle credit-card CSV and drop exact duplicate rows."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLS + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Loaded %s rows (%s after removing duplicates), %s frauds (%.3f%%)",
             f"{n_raw:,}", f"{len(df):,}", int(df["Class"].sum()), df["Class"].mean() * 100)
    df.attrs["rows_raw"] = n_raw
    return df


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological train / validation / test split: models only learn from the past."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]
