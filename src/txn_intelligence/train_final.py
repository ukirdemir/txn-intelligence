"""Train the serving model (gradient boosting, pre-approval features) and save the bundle."""

from __future__ import annotations

import argparse

import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import __version__
from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold, cost_per_1k, expected_cost, pr_auc
from .features import build_features
from .serving import ModelBundle, save_bundle


def fit_bundle(df: pd.DataFrame) -> ModelBundle:
    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")

    X_tr, y_tr = build_features(train, "pre")
    X_va, y_va = build_features(val, "pre")
    X_te, y_te = build_features(test, "pre")

    model = HistGradientBoostingClassifier(
        max_iter=200, class_weight="balanced", random_state=0
    ).fit(X_tr, y_tr)

    s_va = model.predict_proba(X_va)[:, 1]
    s_te = model.predict_proba(X_te)[:, 1]
    threshold, _ = best_threshold(y_va.to_numpy(), s_va, COSTS)

    n_test = len(y_te)
    metrics = {
        "n_test": n_test,
        "test_pr_auc": pr_auc(y_te, s_te),
        "test_cost_per_1k": cost_per_1k(
            expected_cost(y_te.to_numpy(), s_te, threshold, COSTS), n_test
        ),
    }
    return ModelBundle(
        model=model,
        threshold=float(threshold),
        feature_columns=list(X_tr.columns),
        version=f"hgb-pre-{__version__}",
        sklearn_version=sklearn.__version__,
        metrics=metrics,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--out", default="artifacts/model.joblib")
    args = parser.parse_args(argv)

    bundle = fit_bundle(read_transactions(args.db))
    path = save_bundle(bundle, args.out)

    print(f"version     {bundle.version}")
    print(f"threshold   {bundle.threshold:.6f}")
    for key, value in bundle.metrics.items():
        print(f"{key:12s} {value}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
