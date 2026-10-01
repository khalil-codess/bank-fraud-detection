import numpy as np
import pandas as pd
import pytest
from conftest import config_for

from fraud.data import time_split
from fraud.datasets import sparkov
from fraud.models import build_models
from fraud.reasons import format_reasons, group_contributions, reason_codes


@pytest.fixture(scope="module")
def xgb(module_spec):
    spec = module_spec
    train, _, test = time_split(spec.synthetic(4000, seed=0))
    params = config_for(spec.name).models["XGBoost"]
    model = build_models(train["Class"], {"XGBoost": params}, spec.scale_cols, seed=0)["XGBoost"]
    model.fit(spec.add_features(train), train["Class"])
    return spec, model, test


def test_every_feature_belongs_to_a_reason_group(spec):
    assert set(spec.reason_groups) == set(spec.features)


def test_every_group_has_a_sentence(spec):
    df = spec.synthetic(500, seed=1)
    rows = [df.index[0], df.index[-1]]  # includes a card's first transaction and one with history
    feats = spec.add_features(df)
    for group in set(spec.reason_groups.values()):
        for i in rows:
            text = spec.describe(group, df.loc[i], feats.loc[i])
            assert isinstance(text, str) and text and text != group, (group, text)


def test_reasons_are_positive_ranked_and_capped(xgb):
    spec, model, test = xgb
    for reasons in reason_codes(model, test.iloc[:50], spec, top_k=3):
        assert len(reasons) <= 3
        impacts = [r["impact"] for r in reasons]
        assert all(v > 0 for v in impacts)
        assert impacts == sorted(impacts, reverse=True)


def test_groups_add_up_to_the_model_score(xgb):
    """SHAP additivity: base value + every group's contribution = the model's log-odds."""
    spec, model, test = xgb
    rows = test.iloc[:20]
    contrib, base = group_contributions(model, rows, spec)
    p = model.predict_proba(spec.add_features(rows))[:, 1]
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), np.log(p / (1 - p)), atol=1e-3)


def test_frauds_get_reasons(xgb):
    spec, model, test = xgb
    frauds = test[test["Class"] == 1].iloc[:10]
    assert all(reason_codes(model, frauds, spec))
    assert " | " in format_reasons(reason_codes(model, frauds.iloc[[0]], spec)[0])


def test_sparkov_sentences_use_the_transaction_values():
    df = sparkov.synthetic(2000, seed=3)
    row = df[(df["hist_n"] > 3) & (df["hist_amount_24h"] > 0)].iloc[[0]]
    raw, feats = row.iloc[0], sparkov.add_features(row).iloc[0]
    assert f"{raw['Amount']:,.2f}" in sparkov.describe("amount", raw, feats)
    assert f"{raw['hist_amount_24h']:,.2f}" in sparkov.describe("spend_24h", raw, feats)
    assert f"{raw['hist_mean']:,.2f}" in sparkov.describe("amount_vs_usual", raw, feats)


def test_sparkov_hides_the_generator_prefix_in_merchant_names():
    row = sparkov.synthetic(50, seed=4).iloc[[0]].assign(merchant="fraud_Kub Ltd", hist_merchant_n=0)
    text = sparkov.describe("new_merchant", row.iloc[0], sparkov.add_features(row).iloc[0])
    assert text == "First purchase at Kub Ltd"


def test_sparkov_first_transaction_wording():
    first = sparkov.synthetic(300, seed=5)
    first = first[first["hist_n"] == 0].iloc[[0]]
    raw, feats = first.iloc[0], sparkov.add_features(first).iloc[0]
    assert sparkov.describe("recency", raw, feats) == "First transaction seen on this card"
    assert "No earlier" in sparkov.describe("amount_vs_usual", raw, feats)
    assert pd.isna(raw["hist_prev_lat"]) and "No previous" in sparkov.describe("travel", raw, feats)


def test_protected_attributes_are_flagged():
    assert sparkov.SPEC.protected_groups == {"cardholder_age", "cardholder_gender"}
    assert set(sparkov.SPEC.protected_groups) <= set(sparkov.REASON_GROUPS.values())
    reasons = [{"text": "Cardholder age 46", "protected": True},
               {"text": "Amount 5.00 USD", "protected": False}]
    assert format_reasons(reasons) == "Cardholder age 46 [protected attribute] | Amount 5.00 USD"
