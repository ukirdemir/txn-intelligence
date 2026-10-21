"""Bootstrap confidence intervals for test metrics, with paired differences between models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold
from .sensitivity import ALL_MODELS, score_models

METRICS = ("cost_per_1k", "recall", "pr_auc", "alerts_per_1k")
METHODS = ("rows", "days")
DEFAULT_MODELS = ALL_MODELS[1:]
REFERENCE = "hist_gradient_boosting"


def poisson_weights(
    n: int, rng: np.random.Generator, groups: np.ndarray | None = None
) -> np.ndarray:
    """Poisson(1) bootstrap weights per row, or per group of rows when groups are given."""
    if groups is None:
        return rng.poisson(1.0, n).astype(float)
    codes = np.unique(groups, return_inverse=True)[1]
    return rng.poisson(1.0, codes.max() + 1).astype(float)[codes]


def prepare(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precompute everything that does not change between bootstrap replicates."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(scores) >= threshold
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    s = np.asarray(scores, dtype=float)[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    return {
        "tp": (y & flagged).astype(float),
        "fp": (~y & flagged).astype(float),
        "fn": (y & ~flagged).astype(float),
        "order": order,
        "ends": ends,
        "y_sorted": y[order].astype(float),
    }


def weighted_metrics(prep: dict, w: np.ndarray) -> dict:
    """Test metrics at a fixed threshold when each row counts w times."""
    tp, fp, fn, n = w @ prep["tp"], w @ prep["fp"], w @ prep["fn"], w.sum()
    cost = fn * COSTS.false_negative + fp * COSTS.false_positive + tp * COSTS.true_positive

    ws = w[prep["order"]]
    tp_c = np.cumsum(ws * prep["y_sorted"])[prep["ends"]]
    fp_c = np.cumsum(ws * (1 - prep["y_sorted"]))[prep["ends"]]
    total = tp_c[-1]
    denom = tp_c + fp_c
    precision = np.divide(tp_c, denom, out=np.zeros_like(tp_c), where=denom > 0)
    pr = np.sum(np.diff(tp_c / total, prepend=0.0) * precision) if total > 0 else np.nan

    return {
        "cost_per_1k": float(cost / n * 1000),
        "recall": float(tp / (tp + fn)) if tp + fn > 0 else float("nan"),
        "pr_auc": float(pr),
        "alerts_per_1k": float((tp + fp) / n * 1000),
    }


def bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Point estimates and per-replicate metrics; all models share the same weights."""
    preps = {name: prepare(y, scores[name], thresholds[name]) for name in scores}
    n = len(y)
    point = {name: weighted_metrics(p, np.ones(n)) for name, p in preps.items()}
    rng = np.random.default_rng(seed)
    draws = {name: {m: np.empty(n_boot) for m in METRICS} for name in preps}
    for b in range(n_boot):
        w = poisson_weights(n, rng, groups)
        for name, p in preps.items():
            for metric, value in weighted_metrics(p, w).items():
                draws[name][metric][b] = value
    return {"point": point, "draws": draws}


def _interval(values: np.ndarray, point: float) -> dict:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def summarise(result: dict, thresholds: dict[str, float], reference: str) -> dict:
    """95% percentile intervals per model and paired differences against the reference."""
    models = {}
    for name, draws in result["draws"].items():
        models[name] = {"threshold": float(thresholds[name])}
        for metric in METRICS:
            models[name][metric] = _interval(draws[metric], result["point"][name][metric])

    paired = {}
    if reference in result["draws"]:
        ref = result["draws"][reference]
        for name, draws in result["draws"].items():
            if name == reference:
                continue
            cost = draws["cost_per_1k"] - ref["cost_per_1k"]
            recall = ref["recall"] - draws["recall"]
            paired[name] = {
                "cost_other_minus_reference": _interval(
                    cost,
                    result["point"][name]["cost_per_1k"] - result["point"][reference]["cost_per_1k"],
                ),
                "recall_reference_minus_other": _interval(
                    recall, result["point"][reference]["recall"] - result["point"][name]["recall"]
                ),
                "share_reference_cheaper": float(np.mean(cost > 0)),
            }
    return {"models": models, "paired": paired}


def run_bootstrap(
    df,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    method: str = "rows",
    n_boot: int = 1000,
    seed: int = 0,
    epochs: int = 5,
    reference: str = REFERENCE,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    scored = score_models(df, tuple(model_names), epochs)
    thresholds = {
        name: best_threshold(scored["y_val"], scored["scores_val"][name], COSTS)[0]
        for name in model_names
    }
    groups = None
    if method == "days":
        groups = (time_split3(df)[2]["step"].to_numpy() // 24).astype(int)
    result = bootstrap(
        scored["y_test"], scored["scores_test"], thresholds, groups, n_boot, seed
    )
    summary = summarise(result, thresholds, reference)
    return {
        "method": method,
        "n_boot": n_boot,
        "n_test": len(scored["y_test"]),
        "n_test_fraud": int(scored["y_test"].sum()),
        "reference": reference,
        **summary,
    }


def _fmt(cell: dict, digits: int) -> str:
    return f"{cell['point']:.{digits}f} [{cell['lo']:.{digits}f}, {cell['hi']:.{digits}f}]"


def format_table(summary: dict) -> str:
    lines = [
        "| Model | Cost / 1k | Recall | PR-AUC | Alerts / 1k |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        lines.append(
            f"| {name} | {_fmt(m['cost_per_1k'], 1)} | {_fmt(m['recall'], 3)} | "
            f"{_fmt(m['pr_auc'], 3)} | {_fmt(m['alerts_per_1k'], 2)} |"
        )
    if summary["paired"]:
        ref = summary["reference"]
        lines += [
            "",
            f"Paired differences against {ref}:",
            "",
            (
                "| Model | Cost / 1k: model minus reference | Recall: reference minus model "
                "| Share of replicates where reference is cheaper |"
            ),
            "|---|---|---|---|",
        ]
        for name, p in summary["paired"].items():
            lines.append(
                f"| {name} | {_fmt(p['cost_other_minus_reference'], 1)} | "
                f"{_fmt(p['recall_reference_minus_other'], 3)} | "
                f"{p['share_reference_cheaper'] * 100:.1f}% |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=ALL_MODELS)
    parser.add_argument("--method", choices=METHODS, default="rows")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    summary = run_bootstrap(
        read_transactions(args.db),
        model_names=tuple(args.models),
        method=args.method,
        n_boot=args.n_boot,
        seed=args.seed,
        epochs=args.epochs,
    )
    out = Path(args.out or f"artifacts/bootstrap_{args.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_table(summary))
    print(f"\n{summary['n_boot']} replicates, method={summary['method']}, 95% percentile intervals.")
    print("Thresholds are fixed from validation, so intervals cover test sampling only.")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
