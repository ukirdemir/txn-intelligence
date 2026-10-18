"""Cost-ratio sensitivity of thresholds, costs and model ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COSTS, CostMatrix
from .data import read_transactions, time_split3
from .evaluate import best_threshold, expected_cost
from .features import build_features
from .models import RuleBaseline
from .train_compare import RULE_NAMES, make_models

ALL_MODELS = (*RULE_NAMES, "logistic_regression", "hist_gradient_boosting", "torch_mlp")
DEFAULT_MODELS = ALL_MODELS[:4]
DEFAULT_FN_COSTS = (10, 20, 50, 100, 200, 500)


def score_models(
    df: pd.DataFrame,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    epochs: int = 5,
    batch_size: int = 4096,
) -> dict:
    """Fit each model once and keep its validation and test scores."""
    unknown = set(model_names) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")

    X_tr, y_tr = build_features(train, "pre")
    X_va, y_va = build_features(val, "pre")
    X_te, y_te = build_features(test, "pre")

    available = {
        RULE_NAMES[0]: RuleBaseline("exceeds"),
        RULE_NAMES[1]: RuleBaseline("equals"),
        **make_models(epochs, batch_size),
    }
    scores_val, scores_test = {}, {}
    for name in model_names:
        model = available[name]
        model.fit(X_tr, y_tr)
        scores_val[name] = model.predict_proba(X_va)[:, 1]
        scores_test[name] = model.predict_proba(X_te)[:, 1]

    return {
        "y_val": y_va.to_numpy(),
        "y_test": y_te.to_numpy().astype(bool),
        "scores_val": scores_val,
        "scores_test": scores_test,
    }


def sweep(scored: dict, fn_costs: tuple[int, ...] = DEFAULT_FN_COSTS) -> list[dict]:
    """Re-tune each model's threshold on validation for every missed-fraud cost."""
    yv, yt = scored["y_val"], scored["y_test"]
    n_test = len(yt)
    rows = []
    for fn_cost in fn_costs:
        costs = CostMatrix(
            false_negative=float(fn_cost),
            false_positive=COSTS.false_positive,
            true_positive=COSTS.true_positive,
        )
        nothing = expected_cost(yt, np.zeros(n_test), 1.0, costs)
        for name, s_va in scored["scores_val"].items():
            s_te = scored["scores_test"][name]
            threshold, _ = best_threshold(yv, s_va, costs)
            flagged = s_te >= threshold
            total = expected_cost(yt, s_te, threshold, costs)
            rows.append(
                {
                    "fn_cost": float(fn_cost),
                    "ratio": float(fn_cost) / costs.false_positive,
                    "model": name,
                    "threshold": float(threshold),
                    "cost_per_1k": float(total / n_test * 1000),
                    "reduction_vs_nothing_pct": float(100 * (1 - total / nothing)),
                    "alerts_per_1k": float(flagged.sum() / n_test * 1000),
                    "n_missed_fraud": int((yt & ~flagged).sum()),
                }
            )
    return rows


def run_sensitivity(
    df: pd.DataFrame,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    fn_costs: tuple[int, ...] = DEFAULT_FN_COSTS,
    epochs: int = 5,
    batch_size: int = 4096,
) -> dict:
    scored = score_models(df, model_names, epochs, batch_size)
    rows = sweep(scored, fn_costs)
    best = {}
    for fn_cost in fn_costs:
        group = [r for r in rows if r["fn_cost"] == float(fn_cost)]
        best[str(fn_cost)] = min(group, key=lambda r: r["cost_per_1k"])["model"]
    return {
        "false_positive_cost": COSTS.false_positive,
        "n_test": len(scored["y_test"]),
        "n_test_fraud": int(scored["y_test"].sum()),
        "rows": rows,
        "best_model_by_fn_cost": best,
    }


def format_table(summary: dict) -> str:
    lines = [
        (
            "| Missed-fraud cost | Ratio | Model | Cost / 1k | Reduction vs flag nothing "
            "| Alerts / 1k | Missed |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['fn_cost']:.0f} | {r['ratio']:.0f}:1 | {r['model']} | "
            f"{r['cost_per_1k']:.1f} | {r['reduction_vs_nothing_pct']:.1f}% | "
            f"{r['alerts_per_1k']:.2f} | {r['n_missed_fraud']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=ALL_MODELS)
    parser.add_argument("--fn-costs", nargs="+", type=int, default=list(DEFAULT_FN_COSTS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="artifacts/sensitivity_pre.json")
    args = parser.parse_args(argv)

    summary = run_sensitivity(
        read_transactions(args.db),
        model_names=tuple(args.models),
        fn_costs=tuple(args.fn_costs),
        epochs=args.epochs,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_table(summary))
    print(f"\nFalse-alert cost fixed at {summary['false_positive_cost']}.")
    print("Best model by missed-fraud cost:")
    for fn_cost, model in summary["best_model_by_fn_cost"].items():
        print(f"  {fn_cost}: {model}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
