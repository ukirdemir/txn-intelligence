"""Feature sets: "full" (post-transaction balances) and "pre" (known before approval)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRANSACTION_TYPES

_TYPE_COLUMNS = [f"type_{t}" for t in TRANSACTION_TYPES]

FEATURE_COLUMNS = [
    "log_amount",
    "hour_of_day",
    "error_orig",
    "error_dest",
    "orig_emptied",
    *_TYPE_COLUMNS,
]

PRE_FEATURE_COLUMNS = [
    "log_amount",
    "hour_of_day",
    "log_old_balance_orig",
    "log_old_balance_dest",
    "amount_to_orig_balance",
    "orig_would_be_emptied",
    "dest_balance_zero",
    *_TYPE_COLUMNS,
]

FEATURE_SETS = {"full": FEATURE_COLUMNS, "pre": PRE_FEATURE_COLUMNS}


def build_X(df: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Feature matrix only; "pre" works on live requests without labels or new balances."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")

    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log1p(df["amount"])
    X["hour_of_day"] = df["step"] % 24
    for t in TRANSACTION_TYPES:
        X[f"type_{t}"] = (df["type"] == t).astype(int)

    if feature_set == "full":
        X["error_orig"] = df["new_balance_orig"] + df["amount"] - df["old_balance_orig"]
        X["error_dest"] = df["old_balance_dest"] + df["amount"] - df["new_balance_dest"]
        X["orig_emptied"] = (
            (df["new_balance_orig"] == 0) & (df["old_balance_orig"] > 0)
        ).astype(int)
    else:
        old_orig = df["old_balance_orig"].to_numpy(dtype=float)
        amount = df["amount"].to_numpy(dtype=float)
        X["log_old_balance_orig"] = np.log1p(old_orig)
        X["log_old_balance_dest"] = np.log1p(df["old_balance_dest"].to_numpy(dtype=float))
        ratio = np.divide(amount, old_orig, out=np.zeros_like(amount), where=old_orig > 0)
        X["amount_to_orig_balance"] = np.clip(ratio, 0.0, 10.0)
        X["orig_would_be_emptied"] = ((old_orig > 0) & (amount >= old_orig)).astype(int)
        X["dest_balance_zero"] = (df["old_balance_dest"] == 0).astype(int)

    return X[FEATURE_SETS[feature_set]]


def build_features(df: pd.DataFrame, feature_set: str = "full") -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a normalised PaySim DataFrame."""
    return build_X(df, feature_set), df["is_fraud"].astype(int)
