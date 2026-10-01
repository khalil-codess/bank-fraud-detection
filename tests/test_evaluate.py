import numpy as np

from fraud.evaluate import bootstrap_ci, evaluate, select_threshold


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
