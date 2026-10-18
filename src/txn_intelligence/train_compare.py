"""Compare rules, logistic regression, gradient boosting and the MLP on identical splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import (
    best_threshold,
    cost_per_1k,
    expected_cost,
    pr_auc,
    precision_at_recall,
    recall_at_fpr,
)
from .features import build_features
from .models import RuleBaseline, TorchMLP

RULE_NAMES = ("rule_balance_exceeded", "rule_balance_equal")


def make_models(epochs: int, batch_size: int = 4096) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=0
        ),
        "torch_mlp": TorchMLP(epochs=epochs, batch_size=batch_size),
    }


def compare_models(
    df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 4096,
    feature_set: str = "full",
) -> dict:
    """Return per-model test metrics and a simple error analysis."""
    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")

    X_tr, y_tr = build_features(train, feature_set)
    X_va, y_va = build_features(val, feature_set)
    X_te, y_te = build_features(test, feature_set)
    yv, yt = y_va.to_numpy(), y_te.to_numpy().astype(bool)
    n_test = len(yt)
    test_types = test["type"].to_numpy()

    results = []
    missed_masks = []
    models = make_models(epochs, batch_size)
    if feature_set == "pre":
        models = {
            RULE_NAMES[0]: RuleBaseline("exceeds"),
            RULE_NAMES[1]: RuleBaseline("equals"),
            **models,
        }

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        s_va = model.predict_proba(X_va)[:, 1]
        s_te = model.predict_proba(X_te)[:, 1]

        threshold, _ = best_threshold(yv, s_va, COSTS)
        flagged = s_te >= threshold
        missed = yt & ~flagged
        false_alerts = ~yt & flagged
        if name not in RULE_NAMES:
            missed_masks.append(missed)

        total_cost = expected_cost(yt, s_te, threshold, COSTS)
        results.append(
            {
                "model": name,
                "threshold": float(threshold),
                "pr_auc": pr_auc(yt, s_te),
                "recall_at_1pct_fpr": recall_at_fpr(yt, s_te, 0.01),
                "precision_at_90pct_recall": precision_at_recall(yt, s_te, 0.9),
                "cost_per_1k": cost_per_1k(total_cost, n_test),
                "alerts_per_1k": float(flagged.sum() / n_test * 1000),
                "n_missed_fraud": int(missed.sum()),
                "n_false_alerts": int(false_alerts.sum()),
                "missed_fraud_by_type": {
                    str(k): int(v)
                    for k, v in pd.Series(test_types[missed]).value_counts().items()
                },
            }
        )

    missed_by_all = np.logical_and.reduce(missed_masks)
    return {
        "feature_set": feature_set,
        "n_test": n_test,
        "n_test_fraud": int(yt.sum()),
        "cost_per_1k_flag_nothing": cost_per_1k(
            expected_cost(yt, np.zeros(n_test), 1.0, COSTS), n_test
        ),
        "missed_by_all_models": int(missed_by_all.sum()),
        "results": results,
    }


def format_table(summary: dict) -> str:
    lines = [
        "| Model | PR-AUC | Recall @1% FPR | Precision @90% recall | Cost / 1k | Alerts / 1k |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['model']} | {r['pr_auc']:.3f} | {r['recall_at_1pct_fpr']:.3f} | "
            f"{r['precision_at_90pct_recall']:.3f} | {r['cost_per_1k']:.1f} | "
            f"{r['alerts_per_1k']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--features",
        choices=["full", "pre"],
        default="pre",
        help="'pre' uses only information available before approval (realistic, default)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    summary = compare_models(
        read_transactions(args.db), epochs=args.epochs, feature_set=args.features
    )

    out = Path(args.out or f"artifacts/model_comparison_{args.features}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_table(summary))
    print(f"\nCost / 1k if nothing is flagged: {summary['cost_per_1k_flag_nothing']:.1f}")
    print(f"Fraud cases missed by every model: {summary['missed_by_all_models']}"
          f" of {summary['n_test_fraud']}")
    for r in summary["results"]:
        print(f"{r['model']}: missed by type {r['missed_fraud_by_type']}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
