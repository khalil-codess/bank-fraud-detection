import numpy as np
import pytest

from fraud.evaluate import (
    at_k,
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
    savings_curve,
    select_threshold,
    select_threshold_by_cost,
)


def test_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.8])
    m = evaluate(y, p, select_threshold(y, p))
    assert (m["fp"], m["fn"]) == (0, 0)


def test_higher_beta_never_raises_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.2, 0.15, 2000), 0, 1)
    assert select_threshold(y, p, beta=3) <= select_threshold(y, p, beta=1)


def test_confusion_counts():
    m = evaluate(np.array([0, 1, 0, 1, 0]), np.array([0.2, 0.9, 0.7, 0.1, 0.3]), 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)


def test_bootstrap_ci_is_ordered_and_bounded():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    lo, hi = bootstrap_ci(y, y * 0.6 + rng.random(500) * 0.4, n=50)
    assert 0 <= lo <= hi <= 1 + 1e-9


# ── cost model ───────────────────────────────────────────────────────────────

def test_savings_curve_by_hand():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    amount = np.array([100.0, 50.0, 20.0, 10.0])
    thr, n_alerts, savings = savings_curve(y, p, amount, review_cost=5)
    np.testing.assert_array_equal(n_alerts, [1, 2, 3, 4])
    np.testing.assert_allclose(savings, [95, 90, 105, 100])  # caught - 5 * alerts
    assert select_threshold_by_cost(y, p, amount, review_cost=5) == 0.7


def test_tied_scores_are_flagged_together():
    y = np.array([1, 0, 0])
    p = np.array([0.5, 0.5, 0.1])
    _, n_alerts, _ = savings_curve(y, p, np.array([100.0, 1.0, 1.0]), review_cost=1)
    assert list(n_alerts) == [2, 3]


def test_no_alerts_when_reviews_cost_more_than_fraud():
    y = np.array([1, 0, 0])
    p = np.array([0.9, 0.5, 0.1])
    thr = select_threshold_by_cost(y, p, np.array([3.0, 10.0, 10.0]), review_cost=50)
    assert (p >= thr).sum() == 0


def test_cost_metrics_consistent_with_evaluate():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    m = evaluate(y, p, 0.5, amount=np.array([100.0, 50.0, 20.0, 10.0]), review_cost=5)
    assert m["n_alerts"] == m["tp"] + m["fp"] == 2
    assert m["fraud_amount_caught"] == 100
    assert m["savings"] == 90
    assert m["savings_rate"] == pytest.approx(90 / 120)


def test_at_k():
    y = np.array([1, 0, 1, 0, 0])
    r = at_k(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1]), k=2)
    assert (r["precision"], r["recall"], r["frauds_caught"]) == (0.5, 0.5, 1)


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    good = y * 0.7 + rng.random(1000) * 0.3
    bad = rng.random(1000)
    res = paired_bootstrap(y, good, bad, n=200)
    assert res["diff"] > 0
    assert res["ci95"][0] > 0
    assert res["p_a_better"] > 0.99


@pytest.mark.parametrize("ties", [False, True])
def test_fast_average_precision_matches_sklearn(ties):
    from sklearn.metrics import average_precision_score

    from fraud.evaluate import _SortedScores

    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.random(5000), 0, 1)
    if ties:
        p = np.round(p, 1)  # many tied scores
    w = np.bincount(rng.integers(0, 5000, 5000), minlength=5000)
    scores = _SortedScores(y, p)
    assert scores.average_precision() == pytest.approx(average_precision_score(y, p))
    assert scores.average_precision(w) == pytest.approx(average_precision_score(y, p, sample_weight=w))
