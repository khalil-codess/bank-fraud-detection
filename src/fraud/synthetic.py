"""Synthetic transactions in the Kaggle schema, for tests and CI smoke runs.

Usage: python -m fraud.synthetic --rows 5000 --out data/synthetic.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fraud.features import V_COLS


def make_synthetic(n: int = 3000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Random transactions; frauds have shifted V1/V2 so a model can learn them."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = rng.exponential(80, n).round(2)
    df["Class"] = (rng.random(n) < fraud_rate).astype(int)
    df.loc[df["Class"] == 1, ["V1", "V2"]] += 4
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic transactions CSV.")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    make_synthetic(args.rows, args.fraud_rate, args.seed).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
