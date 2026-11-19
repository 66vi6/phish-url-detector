import numpy as np
from src.metrics import recall_at_fpr, precision_at_prevalence
from src.metrics import threshold_at_fpr, wilson, cluster_bootstrap


def test_perfect_separation():
    y = np.array([0] * 1000 + [1] * 100)
    p = np.array([0.1] * 1000 + [0.9] * 100)
    assert recall_at_fpr(y, p, 0.001) == 1.0
    assert precision_at_prevalence(y, p, 0.001, 0.01) == 1.0


def test_random_scores_have_low_recall_at_low_fpr():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = rng.random(5000)
    assert recall_at_fpr(y, p, 0.01) < 0.1

def test_threshold_respects_fpr():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = rng.random(4000) + 0.3 * y
    thr = threshold_at_fpr(y, p, 0.05)
    assert (p[y == 0] >= thr).mean() <= 0.05 + 1e-9


def test_wilson_contains_point_estimate():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_cluster_bootstrap_perfect_scores():
    y = np.array([0] * 200 + [1] * 100)
    groups = np.arange(300) // 2                     # pure pairs, never mixing classes
    res = cluster_bootstrap(y, {"m": y.astype(float)}, groups, n_boot=20, seed=0)
    assert res["m"]["ap"].min() == 1.0