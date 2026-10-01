"""Synthetic transactions in a dataset's schema, for tests, CI and trying the project without Kaggle.

Usage: python -m fraud.synthetic --dataset sparkov --rows 20000 --out data/synthetic_sparkov.csv
"""

from __future__ import annotations

import argparse

from fraud.datasets import DATASETS, get_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic transactions CSV.")
    ap.add_argument("--dataset", choices=DATASETS, default="sparkov")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = get_dataset(args.dataset)
    # written in the dataset's own file format, so the real loader reads it
    spec.to_raw(spec.synthetic(args.rows, args.fraud_rate, args.seed)).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
