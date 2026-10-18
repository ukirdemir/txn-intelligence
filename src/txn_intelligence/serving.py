"""Model bundle and single-transaction scoring. No torch; load only bundles you created."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .features import build_X


@dataclass
class ModelBundle:
    model: object
    threshold: float
    feature_columns: list[str]
    version: str
    sklearn_version: str
    metrics: dict = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if bundle.sklearn_version != sklearn.__version__:
        warnings.warn(
            f"Bundle was trained with scikit-learn {bundle.sklearn_version} but "
            f"{sklearn.__version__} is installed; predictions may differ or fail.",
            stacklevel=2,
        )
    return bundle


def transaction_frame(txn: dict) -> pd.DataFrame:
    """One-row DataFrame with the pre-approval fields the feature builder needs."""
    cols = ("step", "type", "amount", "old_balance_orig", "old_balance_dest")
    return pd.DataFrame([{c: txn[c] for c in cols}])


def rule_flag(txn: dict) -> bool:
    """Simple benchmark: TRANSFER/CASH_OUT where the amount equals the sender's balance."""
    return bool(
        txn["type"] in ("TRANSFER", "CASH_OUT")
        and txn["old_balance_orig"] > 0
        and txn["amount"] == txn["old_balance_orig"]
    )


def score_transaction(bundle: ModelBundle, txn: dict) -> dict:
    X = build_X(transaction_frame(txn), "pre")[bundle.feature_columns]
    p = float(bundle.model.predict_proba(X)[0, 1])
    return {
        "fraud_score": p,
        "flagged": bool(p >= bundle.threshold),
        "threshold": float(bundle.threshold),
        "rule_flag": rule_flag(txn),
        "model_version": bundle.version,
    }
