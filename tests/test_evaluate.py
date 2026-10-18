import numpy as np

from txn_intelligence.config import CostMatrix
from txn_intelligence.data import make_synthetic, time_split
from txn_intelligence.evaluate import best_threshold, expected_cost

COSTS = CostMatrix(false_negative=100.0, false_positive=5.0, true_positive=1.0)


def test_expected_cost_hand_computed():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1])
    assert expected_cost(y, s, 0.5, COSTS) == 100.0 + 5.0 + 1.0


def test_best_threshold_perfect_separation():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
    t, cost = best_threshold(y, s, COSTS)
    assert cost == 2.0
    assert 0.2 < t <= 0.8


def test_time_split_has_no_overlap_in_time():
    df = make_synthetic(n=1000)
    train, test = time_split(df)
    assert train["step"].max() <= test["step"].min()
