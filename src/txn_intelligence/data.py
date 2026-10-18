"""PaySim loading, validation, DuckDB helpers and time-based splits."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PAYSIM_RENAME = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def normalize_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw PaySim columns, rename to snake_case and check basic sanity."""
    missing = set(PAYSIM_RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"Missing PaySim columns: {sorted(missing)}")

    out = df.rename(columns=PAYSIM_RENAME)[list(PAYSIM_RENAME.values())].copy()

    if not out["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0/1 values")
    if (out["amount"] < 0).any():
        raise ValueError("amount must be non-negative")
    unknown = set(out["type"].unique()) - set(TRANSACTION_TYPES)
    if unknown:
        raise ValueError(f"Unknown transaction types: {sorted(unknown)}")

    out["is_fraud"] = out["is_fraud"].astype(int)
    out["is_flagged_fraud"] = out["is_flagged_fraud"].astype(int)
    return out


def load_paysim(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw PaySim CSV and return a validated, snake_case DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download steps.")
    return normalize_paysim(pd.read_csv(path))


def write_duckdb(df: pd.DataFrame, db_path: str | Path, table: str = "transactions") -> Path:
    """Persist a DataFrame as a DuckDB table so the SQL files in sql/ can run on it."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("df_view", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_view")
    finally:
        con.close()
    return db_path


def run_sql_file(db_path: str | Path, sql_path: str | Path) -> pd.DataFrame:
    """Execute a single-statement SQL file against a DuckDB database."""
    sql = Path(sql_path).read_text(encoding="utf-8")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def time_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time (not randomly) to avoid leaking the future into training."""
    df = df.sort_values("step", kind="stable")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def read_transactions(db_path: str | Path, table: str = "transactions") -> pd.DataFrame:
    """Load the prepared transactions table from DuckDB."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table}").df()
    finally:
        con.close()


def time_split3(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/validation/test split in time order. Thresholds are tuned on validation only."""
    df = df.sort_values("step", kind="stable")
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def make_synthetic(n: int = 10_000, fraud_rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic transaction table. Fraud tends to have larger amounts."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=5.0, sigma=1.0, size=n),
        rng.lognormal(mean=3.5, sigma=1.0, size=n),
    )
    return pd.DataFrame(
        {
            "step": np.sort(rng.integers(0, 720, size=n)),
            "amount": amount,
            "is_fraud": is_fraud.astype(int),
        }
    )
