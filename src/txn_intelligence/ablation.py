"""Feature ablation: retrain with feature groups removed and compare scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold, cost_per_1k, expected_cost, pr_auc, precision_at_recall
from .features import PRE_FEATURE_COLUMNS, build_features
from .train_compare import make_models

TYPE_COLS = [c for c in PRE_FEATURE_COLUMNS if c.startswith("type_")]

# variant -> columns removed
ABLATIONS: dict[str, list[str]] = {
    "reference": [],
    "drop_account_emptying": ["orig_would_be_emptied", "amount_to_orig_balance"],
    "drop_destination_info": ["dest_balance_zero", "log_old_balance_dest"],
    "drop_origin_balance_info": [
        "log_old_balance_orig",
        "amount_to_orig_balance",
        "orig_would_be_emptied",
    ],
    "drop_type": TYPE_COLS,
    "minimal": [
        c for c in PRE_FEATURE_COLUMNS if c not in ("log_amount", "hour_of_day", *TYPE_COLS)
    ],
}

DEFAULT_MODELS = ("logistic_regression", "hist_gradient_boosting")


def permutation_importance_pr_auc(
    model, X: pd.DataFrame, y: pd.Series, max_rows: int = 200_000, seed: int = 0
) -> dict[str, float]:
    """Drop in PR-AUC when one column is shuffled (larger = more important)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    yv = y.to_numpy()
    if yv.sum() == 0:
        raise ValueError("Permutation sample contains no fraud cases")

    base = pr_auc(yv, model.predict_proba(X)[:, 1])
    out: dict[str, float] = {}
    for col in X.columns:
        shuffled = X.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        out[col] = float(base - pr_auc(yv, model.predict_proba(shuffled)[:, 1]))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def run_ablation(
    df: pd.DataFrame,
    models: tuple[str, ...] = DEFAULT_MODELS,
    epochs: int = 5,
    batch_size: int = 4096,
    permutation: bool = False,
    max_perm_rows: int = 200_000,
) -> dict:
    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")

    X_tr, y_tr = build_features(train, "pre")
    X_va, y_va = build_features(val, "pre")
    X_te, y_te = build_features(test, "pre")
    yv, yt = y_va.to_numpy(), y_te.to_numpy().astype(bool)
    n_test = len(yt)

    results = []
    reference_models: dict = {}
    for variant, dropped in ABLATIONS.items():
        cols = [c for c in PRE_FEATURE_COLUMNS if c not in dropped]
        if not cols:
            raise ValueError(f"Variant {variant} removes every feature")

        fresh = make_models(epochs, batch_size)
        for name in models:
            model = fresh[name]
            model.fit(X_tr[cols], y_tr)
            s_va = model.predict_proba(X_va[cols])[:, 1]
            s_te = model.predict_proba(X_te[cols])[:, 1]

            threshold, _ = best_threshold(yv, s_va, COSTS)
            flagged = s_te >= threshold
            total_cost = expected_cost(yt, s_te, threshold, COSTS)
            results.append(
                {
                    "variant": variant,
                    "model": name,
                    "n_features": len(cols),
                    "features": cols,
                    "dropped": list(dropped),
                    "pr_auc": pr_auc(yt, s_te),
                    "precision_at_90pct_recall": precision_at_recall(yt, s_te, 0.9),
                    "cost_per_1k": cost_per_1k(total_cost, n_test),
                    "alerts_per_1k": float(flagged.sum() / n_test * 1000),
                }
            )
            if variant == "reference":
                reference_models[name] = model

    perm = None
    if permutation:
        name = "hist_gradient_boosting" if "hist_gradient_boosting" in models else models[0]
        perm = {
            "model": name,
            "pr_auc_drop": permutation_importance_pr_auc(
                reference_models[name], X_te, y_te, max_rows=max_perm_rows
            ),
        }

    return {
        "n_test": n_test,
        "n_test_fraud": int(yt.sum()),
        "cost_per_1k_flag_nothing": cost_per_1k(
            expected_cost(yt, np.zeros(n_test), 1.0, COSTS), n_test
        ),
        "results": results,
        "permutation_importance": perm,
    }


def format_table(summary: dict) -> str:
    lines = [
        "| Variant | Model | Features | PR-AUC | Precision @90% recall | Cost / 1k |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['variant']} | {r['model']} | {r['n_features']} | {r['pr_auc']:.3f} | "
            f"{r['precision_at_90pct_recall']:.3f} | {r['cost_per_1k']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=["logistic_regression", "hist_gradient_boosting", "torch_mlp"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--out", default="artifacts/ablation_pre.json")
    args = parser.parse_args(argv)

    summary = run_ablation(
        read_transactions(args.db),
        models=tuple(args.models),
        epochs=args.epochs,
        permutation=args.permutation,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_table(summary))
    print(f"\nCost / 1k if nothing is flagged: {summary['cost_per_1k_flag_nothing']:.1f}")
    if summary["permutation_importance"]:
        p = summary["permutation_importance"]
        print(f"\nPermutation importance ({p['model']}, drop in PR-AUC):")
        for col, drop in p["pr_auc_drop"].items():
            print(f"  {col:28s} {drop:.4f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
