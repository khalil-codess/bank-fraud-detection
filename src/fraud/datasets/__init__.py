"""Dataset registry.

Every dataset is loaded into the same canonical frame:
  - `Time`   seconds since a fixed epoch (used to sort and split chronologically)
  - `Amount` transaction amount (used by the cost model)
  - `Class`  1 = fraud, 0 = legitimate
plus the raw columns its feature function needs (`raw_cols`). Splitting, cross-validation, the cost
model and the dashboard only rely on the canonical columns, so they work for every dataset.

Datasets with card history also define `enrich`, which adds history columns that depend only on
earlier transactions. It runs once on the full frame, before splitting: a test transaction may use
history from the training period, exactly as it would in production.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    raw_cols: list[str]          # columns needed to score one transaction (Time and Amount included)
    features: list[str]          # model input columns, in order
    scale_cols: list[str]        # features standardised by the model pipeline
    load: Callable[[Path], pd.DataFrame]
    add_features: Callable[[pd.DataFrame], pd.DataFrame]
    synthetic: Callable[..., pd.DataFrame]
    # canonical frame -> the file format `load` reads (used to write synthetic CSVs)
    to_raw: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
    # adds point-in-time history columns computed over the whole chronological frame (see history.py)
    enrich: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
    # reason codes (see reasons.py): feature -> reason group, and (group, raw row, feature row) -> text
    reason_groups: dict = field(default_factory=dict)
    describe: Callable[[str, pd.Series, pd.Series], str] = lambda group, raw, feats: group


def get_dataset(name: str) -> DatasetSpec:
    from fraud.datasets import creditcard, sparkov

    registry = {creditcard.SPEC.name: creditcard.SPEC, sparkov.SPEC.name: sparkov.SPEC}
    if name not in registry:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {sorted(registry)}")
    return registry[name]


DATASETS = ("creditcard", "sparkov")
