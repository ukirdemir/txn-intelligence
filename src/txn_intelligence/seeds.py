"""Repeat training with different seeds to measure how much results move from retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold, cost_per_1k, expected_cost, pr_auc
from .features import build_features
from .models import TorchMLP

MODELS = ("hist_gradient_boosting", "torch_mlp")
METRICS = ("cost_per_1k", "recall", "pr_auc", "alerts_per_1k", "n_missed_fraud")


def make_model(name: str, seed: int, epochs: int = 5, batch_size: int = 4096):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=seed
        )
    if name == "torch_mlp":
        return TorchMLP(epochs=epochs, batch_size=batch_size, seed=seed)
    raise ValueError(f"Unknown model: {name}")


def aggregate(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_run(model, X_va, y_va, X_te, y_te) -> dict:
    """Tune the threshold on validation and report test metrics."""
    s_va = model.predict_proba(X_va)[:, 1]
    s_te = model.predict_proba(X_te)[:, 1]
    threshold, _ = best_threshold(y_va.to_numpy(), s_va, COSTS)
    y = y_te.to_numpy().astype(bool)
    flagged = s_te >= threshold
    n = len(y)
    tp = int((y & flagged).sum())
    missed = int((y & ~flagged).sum())
    return {
        "threshold": float(threshold),
        "cost_per_1k": cost_per_1k(expected_cost(y, s_te, threshold, COSTS), n),
        "recall": tp / (tp + missed) if tp + missed else float("nan"),
        "pr_auc": pr_auc(y, s_te),
        "alerts_per_1k": float(flagged.sum() / n * 1000),
        "n_missed_fraud": missed,
    }


def run_seeds(
    df,
    model_names: tuple[str, ...] = MODELS,
    n_seeds: int = 10,
    epochs: int = 5,
    batch_size: int = 4096,
    verbose: bool = False,
) -> dict:
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    train, val, test = time_split3(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part["is_fraud"].sum() == 0:
            raise ValueError(f"The {name} split contains no fraud cases")
    X_tr, y_tr = build_features(train, "pre")
    X_va, y_va = build_features(val, "pre")
    X_te, y_te = build_features(test, "pre")

    runs = {name: [] for name in model_names}
    for seed in range(n_seeds):
        for name in model_names:
            model = make_model(name, seed, epochs, batch_size).fit(X_tr, y_tr)
            result = {"seed": seed, **evaluate_run(model, X_va, y_va, X_te, y_te)}
            runs[name].append(result)
            if verbose:
                print(f"seed {seed} {name}: cost/1k {result['cost_per_1k']:.2f}", flush=True)

    models = {
        name: {"runs": rows, **{m: aggregate([r[m] for r in rows]) for m in METRICS}}
        for name, rows in runs.items()
    }
    return {"n_seeds": n_seeds, "n_test": len(y_te), "models": models}


def _cell(stats: dict, digits: int) -> str:
    return (
        f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f} "
        f"[{stats['min']:.{digits}f}, {stats['max']:.{digits}f}]"
    )


def format_table(summary: dict) -> str:
    lines = [
        "| Model | Cost / 1k | Recall | PR-AUC | Alerts / 1k | Missed fraud |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        lines.append(
            f"| {name} | {_cell(m['cost_per_1k'], 2)} | {_cell(m['recall'], 4)} | "
            f"{_cell(m['pr_auc'], 4)} | {_cell(m['alerts_per_1k'], 2)} | "
            f"{_cell(m['n_missed_fraud'], 1)} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="artifacts/seed_stability.json")
    args = parser.parse_args(argv)

    summary = run_seeds(
        read_transactions(args.db),
        model_names=tuple(args.models),
        n_seeds=args.seeds,
        epochs=args.epochs,
        verbose=True,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(format_table(summary))
    print(f"\n{summary['n_seeds']} seeds; cells are mean ± std [min, max]. Split is fixed in time.")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
