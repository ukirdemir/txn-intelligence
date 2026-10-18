import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def paysim_df() -> pd.DataFrame:
    """Normalised PaySim-like frame where fraud drains the origin account (learnable)."""
    rng = np.random.default_rng(0)
    n = 4000
    is_fraud = rng.random(n) < 0.05
    amount = rng.lognormal(4.0, 1.0, n)
    old_orig = np.where(is_fraud, amount, amount + rng.uniform(1, 1000, n))
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
