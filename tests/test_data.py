from pathlib import Path

import pandas as pd
import pytest

from txn_intelligence.data import (
    PAYSIM_RENAME,
    normalize_paysim,
    run_sql_file,
    write_duckdb,
)
from txn_intelligence.features import FEATURE_COLUMNS, build_features

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def raw_paysim() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 25, 26],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"],
            "amount": [10.0, 500.0, 500.0, 20.0],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "oldbalanceOrg": [100.0, 500.0, 500.0, 50.0],
            "newbalanceOrig": [90.0, 0.0, 0.0, 70.0],
            "nameDest": ["M1", "C9", "C8", "C7"],
            "oldbalanceDest": [0.0, 0.0, 100.0, 10.0],
            "newbalanceDest": [0.0, 500.0, 600.0, 0.0],
            "isFraud": [0, 1, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )


def test_normalize_renames_columns():
    out = normalize_paysim(raw_paysim())
    assert list(out.columns) == list(PAYSIM_RENAME.values())
    assert out["is_fraud"].sum() == 2


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing PaySim columns"):
        normalize_paysim(raw_paysim().drop(columns=["amount"]))


def test_normalize_rejects_unknown_type():
    df = raw_paysim()
    df.loc[0, "type"] = "BITCOIN"
    with pytest.raises(ValueError, match="Unknown transaction types"):
        normalize_paysim(df)


def test_features_shape_and_values():
    X, y = build_features(normalize_paysim(raw_paysim()))
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 4
    assert X["orig_emptied"].tolist() == [0, 1, 1, 0]
    assert X["error_orig"].iloc[0] == 0.0
    assert X["type_TRANSFER"].tolist() == [0, 1, 0, 0]


def test_sql_files_run_on_duckdb(tmp_path):
    db = write_duckdb(normalize_paysim(raw_paysim()), tmp_path / "t.duckdb")
    by_type = run_sql_file(db, SQL_DIR / "01_fraud_rate_by_type.sql")
    assert by_type["n_fraud"].sum() == 2
    by_day = run_sql_file(db, SQL_DIR / "02_fraud_rate_by_day.sql")
    assert by_day["n_txns"].sum() == 4
    emptied = run_sql_file(db, SQL_DIR / "03_fraud_account_emptied.sql")
    fraud_row = emptied[emptied["is_fraud"] == 1].iloc[0]
    assert fraud_row["pct_origin_emptied"] == 100.0
