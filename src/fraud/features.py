"""Feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = V_COLS + ["Time", "Amount"]
FEATURES = V_COLS + ["Amount_log", "Hour_sin", "Hour_cos"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns (V1-V28, Time, Amount) into model features.

    Nothing is learned here, so training and serving compute identical values.
    """
    out = df[V_COLS].copy()
    out["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    # Time = seconds since the first transaction in the file -> position in the 24 h cycle
    hour = (df["Time"] // 3600) % 24
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURES]
