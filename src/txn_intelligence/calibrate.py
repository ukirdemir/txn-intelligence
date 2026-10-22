"""Calibration analysis: isotonic and Platt maps from score to fraud probability, checked on test."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import COSTS
from .data import read_transactions, time_split3
from .evaluate import best_threshold, cost_per_1k, expected_cost
from .sensitivity import score_models

MODEL = "hist_gradient_boosting"
BIN_EDGES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
EPS = 1e-12

Mapper = Callable[[np.ndarray], np.ndarray]


def theory_threshold() -> float:
    """Probability above which flagging is cheaper than not flagging, from the cost matrix."""
    return COSTS.false_positive / (
        COSTS.false_positive + COSTS.false_negative - COSTS.true_positive
    )


def fit_isotonic(y: np.ndarray, scores: np.ndarray) -> Mapper:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
    return lambda s: model.predict(np.asarray(s, dtype=float))


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(y: np.ndarray, scores: np.ndarray) -> Mapper:
    model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores), np.asarray(y).astype(int))
    return lambda s: model.predict_proba(_logit(s))[:, 1]


def reliability(y: np.ndarray, p: np.ndarray, edges: tuple[float, ...] = BIN_EDGES) -> list[dict]:
    """Mean predicted probability against observed fraud rate in each probability bin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        mask = idx == k
        if mask.any():
            close = "]" if k == len(edges) - 2 else ")"
            rows.append(
                {
                    "bin": f"[{edges[k]:g}, {edges[k + 1]:g}{close}",
                    "n": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    return rows


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier score, log loss and expected calibration error over the fixed bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    clipped = np.clip(p, EPS, 1 - EPS)
    n = len(y)
    ece = sum(
        b["n"] / n * abs(b["mean_predicted"] - b["observed_rate"]) for b in reliability(y, p)
    )
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece": float(ece),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def decision(y_val, p_val, y_test, p_test, threshold: float | None = None) -> dict:
    """Cost and alerts on test at a threshold tuned on validation, or at a given one."""
    if threshold is None:
        threshold, _ = best_threshold(y_val, p_val, COSTS)
    y_test = np.asarray(y_test).astype(bool)
    flagged = np.asarray(p_test) >= threshold
    n = len(y_test)
    return {
        "threshold": float(threshold),
        "cost_per_1k": cost_per_1k(expected_cost(y_test, p_test, threshold, COSTS), n),
        "alerts_per_1k": float(flagged.sum() / n * 1000),
        "n_missed_fraud": int((y_test & ~flagged).sum()),
    }


def period_groups(steps: np.ndarray, volume_split: int) -> dict[str, np.ndarray]:
    """Row masks for days with at least volume_split rows and for thinner days."""
    days = np.asarray(steps) // 24
    values, counts = np.unique(days, return_counts=True)
    mask = np.isin(days, values[counts >= volume_split])
    groups = {"high_volume_days": mask, "thin_days": ~mask}
    return {name: m for name, m in groups.items() if m.any()}


def run_calibration(df, volume_split: int = 100_000) -> dict:
    scored = score_models(df, (MODEL,))
    y_val, y_test = scored["y_val"].astype(bool), scored["y_test"].astype(bool)
    s_val, s_test = scored["scores_val"][MODEL], scored["scores_test"][MODEL]

    mappers = {
        "raw_score": lambda s: np.asarray(s, dtype=float),
        "isotonic": fit_isotonic(y_val, s_val),
        "platt": fit_platt(y_val, s_val),
    }
    p_val = {name: m(s_val) for name, m in mappers.items()}
    p_test = {name: m(s_test) for name, m in mappers.items()}

    methods = {
        name: {
            **calibration_metrics(y_test, p_test[name]),
            "reliability": reliability(y_test, p_test[name]),
        }
        for name in mappers
    }
    decisions = {name: decision(y_val, p_val[name], y_test, p_test[name]) for name in mappers}
    decisions["isotonic_at_theory_threshold"] = decision(
        y_val, p_val["isotonic"], y_test, p_test["isotonic"], theory_threshold()
    )

    steps = time_split3(df)[2]["step"].to_numpy()
    periods = {}
    for label, mask in period_groups(steps, volume_split).items():
        periods[label] = {
            "n": int(mask.sum()),
            "fraud_rate": float(y_test[mask].mean()),
            "methods": {
                name: calibration_metrics(y_test[mask], p_test[name][mask]) for name in mappers
            },
        }
    return {
        "model": MODEL,
        "n_val": len(y_val),
        "val_fraud_rate": float(y_val.mean()),
        "n_test": len(y_test),
        "test_fraud_rate": float(y_test.mean()),
        "theory_threshold": theory_threshold(),
        "methods": methods,
        "decisions": decisions,
        "periods": periods,
    }


def format_table(summary: dict) -> str:
    lines = [
        "| Method | Brier | Log loss | ECE | Mean predicted | Observed rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | {m['brier']:.6f} | {m['log_loss']:.5f} | {m['ece']:.5f} | "
            f"{m['mean_predicted']:.5f} | {m['observed_rate']:.5f} |"
        )
    lines += [
        "",
        "| Decision rule | Threshold | Cost / 1k | Alerts / 1k | Missed fraud |",
        "|---|---|---|---|---|",
    ]
    for name, d in summary["decisions"].items():
        lines.append(
            f"| {name} | {d['threshold']:.4f} | {d['cost_per_1k']:.1f} | "
            f"{d['alerts_per_1k']:.2f} | {d['n_missed_fraud']} |"
        )
    lines += [
        "",
        "| Period | Rows | Fraud rate | Method | Mean predicted | Observed rate | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, p in summary["periods"].items():
        for name, m in p["methods"].items():
            lines.append(
                f"| {label} | {p['n']} | {p['fraud_rate']:.5f} | {name} | "
                f"{m['mean_predicted']:.5f} | {m['observed_rate']:.5f} | {m['brier']:.6f} |"
            )
    lines += [
        "",
        "Reliability, isotonic (test):",
        "",
        "| Bin | Rows | Mean predicted | Observed rate |",
        "|---|---|---|---|",
    ]
    for b in summary["methods"]["isotonic"]["reliability"]:
        lines.append(
            f"| {b['bin']} | {b['n']} | {b['mean_predicted']:.4f} | {b['observed_rate']:.4f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--volume-split", type=int, default=100_000)
    parser.add_argument("--out", default="artifacts/calibration.json")
    args = parser.parse_args(argv)

    summary = run_calibration(read_transactions(args.db), args.volume_split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_table(summary))
    print(f"\nTheory threshold from the cost matrix: {summary['theory_threshold']:.4f}")
    print(
        f"Validation fraud rate {summary['val_fraud_rate']:.5f}, "
        f"test {summary['test_fraud_rate']:.5f}."
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
