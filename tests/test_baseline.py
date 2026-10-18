import numpy as np
import pandas as pd
import pytest

from txn_intelligence.config import CostMatrix
from txn_intelligence.evaluate import best_threshold, expected_cost, pr_auc, recall_at_fpr
from txn_intelligence.train_baseline import fit_and_evaluate


def fake_paysim(n: int = 4000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Normalised PaySim-like frame where fraud drains the origin account (learnable)."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    amount = rng.lognormal(4.0, 1.0, n)
    old_orig = amount + rng.uniform(0, 1000, n)
    old_dest = rng.uniform(0, 1000, n)
    types = rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], n)
    types = np.where(is_fraud, rng.choice(["TRANSFER", "CASH_OUT"], n), types)
    return pd.DataFrame(
        {
            "step": np.sort(rng.integers(1, 300, n)),
            "type": types,
            "amount": amount,
            "name_orig": "C0",
            "old_balance_orig": old_orig,
            "new_balance_orig": np.where(is_fraud, 0.0, old_orig - amount),
            "name_dest": "C1",
            "old_balance_dest": old_dest,
            "new_balance_dest": np.where(is_fraud, old_dest, old_dest + amount),
            "is_fraud": is_fraud.astype(int),
            "is_flagged_fraud": 0,
        }
    )


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    y = rng.random(300) < 0.1
    s = rng.random(300) + 0.5 * y
    costs = CostMatrix(false_negative=100.0, false_positive=5.0, true_positive=1.0)
    _, fast_cost = best_threshold(y, s, costs)
    brute = min(
        [expected_cost(y, s, float(t), costs) for t in np.unique(s)]
        + [expected_cost(y, s, np.inf, costs)]
    )
    assert fast_cost == pytest.approx(brute)


def test_metrics_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    assert pr_auc(y, s) == pytest.approx(1.0)
    assert recall_at_fpr(y, s, 0.01) == pytest.approx(1.0)


def test_fit_and_evaluate_beats_trivial_policies():
    res = fit_and_evaluate(fake_paysim())
    assert res["pr_auc"] > 0.9
    assert res["cost_per_1k_model"] < res["cost_per_1k_flag_nothing"]
    assert res["cost_per_1k_model"] < res["cost_per_1k_flag_all"]


def test_fit_and_evaluate_requires_fraud_in_every_split():
    df = fake_paysim()
    df["is_fraud"] = 0
    with pytest.raises(ValueError, match="no fraud"):
        fit_and_evaluate(df)
