import numpy as np
from src.metrics import recall_at_fpr, precision_at_prevalence


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