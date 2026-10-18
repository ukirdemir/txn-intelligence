"""Logistic-regression baseline with cost-aware evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold, cost_per_1k, expected_cost, pr_auc, recall_at_fpr
from .features import build_features


def fit_and_evaluate(df: pd.DataFrame) -> dict:
    """Train on the train split, tune the threshold on validation, report on test."""
    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")

    X_tr, y_tr = build_features(train)
    X_va, y_va = build_features(val)
    X_te, y_te = build_features(test)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)

    s_va = model.predict_proba(X_va)[:, 1]
    s_te = model.predict_proba(X_te)[:, 1]

    threshold, _ = best_threshold(y_va.to_numpy(), s_va, COSTS)
    test_cost = expected_cost(y_te.to_numpy(), s_te, threshold, COSTS)
    flag_nothing = expected_cost(y_te.to_numpy(), s_te, np.inf, COSTS)
    flag_all = expected_cost(y_te.to_numpy(), s_te, -np.inf, COSTS)
    n_test = len(y_te)

    return {
        "model": "logistic_regression_balanced",
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "n_test": n_test,
        "test_fraud_rate": float(y_te.mean()),
        "threshold": float(threshold),
        "pr_auc": pr_auc(y_te, s_te),
        "recall_at_1pct_fpr": recall_at_fpr(y_te, s_te, 0.01),
        "cost_per_1k_model": cost_per_1k(test_cost, n_test),
        "cost_per_1k_flag_nothing": cost_per_1k(flag_nothing, n_test),
        "cost_per_1k_flag_all": cost_per_1k(flag_all, n_test),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--out", default="artifacts/baseline_metrics.json")
    args = parser.parse_args(argv)

    results = fit_and_evaluate(read_transactions(args.db))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for key, value in results.items():
        print(f"{key:28s} {value}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
