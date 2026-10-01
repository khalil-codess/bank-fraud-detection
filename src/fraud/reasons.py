"""Reason codes: the top factors behind a fraud score, in plain language.

SHAP gives one contribution per model feature (in log-odds). Related features are summed into
reason groups (e.g. Hour_sin + Hour_cos -> "time of day"; the 14 category one-hots -> "merchant
category"), groups pushing the score towards fraud are ranked, and each dataset turns the top
groups into a sentence built from the transaction's own values, e.g.
"Card spent 1,212 USD in the previous 24 h".

SHAP values are additive, so the group contributions plus the base value equal the model's raw
log-odds score: the reasons account for the whole score, nothing is hidden.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from fraud.datasets import DatasetSpec
from fraud.inference import explain


def group_contributions(model, raw_df: pd.DataFrame, spec: DatasetSpec):
    """Per-row SHAP contributions summed by reason group -> (DataFrame rows x groups, base values)."""
    expl = explain(model, raw_df, spec)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(expl.feature_names):
        groups[spec.reason_groups[name]].append(i)
    values = np.asarray(expl.values)
    contrib = pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in groups.items()}, index=raw_df.index)
    return contrib, np.asarray(expl.base_values, dtype=float).reshape(-1)


def reason_codes(model, raw_df: pd.DataFrame, spec: DatasetSpec, top_k: int = 3) -> list[list[dict]]:
    """For each row, up to `top_k` reasons that push the score towards fraud, strongest first.

    Each reason: {"group", "impact" (log-odds contribution, > 0), "text", "protected" (True when
    the reason is a protected attribute such as age or gender)}.
    """
    contrib, _ = group_contributions(model, raw_df, spec)
    features = spec.add_features(raw_df)
    out = []
    for idx, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_k)
        raw, feats = raw_df.loc[idx], features.loc[idx]
        out.append([{"group": g, "impact": float(v), "text": spec.describe(g, raw, feats),
                     "protected": g in spec.protected_groups}
                    for g, v in top.items()])
    return out


def format_reasons(reasons: list[dict]) -> str:
    """One-line summary for tables and CSV exports."""
    return " | ".join(r["text"] + (" [protected attribute]" if r.get("protected") else "") for r in reasons)
