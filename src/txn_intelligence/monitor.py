"""Label-free drift monitoring: feature, score and alert-rate drift against a reference period."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import MODEL_PATH
from .data import read_transactions, time_split3
from .features import build_X
from .serving import ModelBundle, load_bundle

EPS = 1e-4
WATCH, INVESTIGATE = 0.1, 0.25
STATUS_ORDER = {"stable": 0, "watch": 1, "investigate": 2}


def _proportions(counts: np.ndarray) -> np.ndarray:
    return np.clip(counts / counts.sum(), EPS, None)


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index; few-valued columns are compared per value, others by quantile."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("psi needs non-empty samples")
    values = np.unique(ref)
    if len(values) <= bins:
        r = np.array([(ref == v).sum() for v in values] + [0])
        c = np.array([(cur == v).sum() for v in values] + [0])
        c[-1] = len(cur) - c[:-1].sum()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0]
        c = np.histogram(cur, edges)[0]
    p, q = _proportions(r), _proportions(c)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_stat(reference, current, max_n: int = 200_000, seed: int = 0) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (effect size, not a p-value)."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) > max_n:
        ref = rng.choice(ref, max_n, replace=False)
    if len(cur) > max_n:
        cur = rng.choice(cur, max_n, replace=False)
    return float(ks_2samp(ref, cur).statistic)


def status(value: float) -> str:
    if value < WATCH:
        return "stable"
    return "watch" if value < INVESTIGATE else "investigate"


def profile(df: pd.DataFrame, bundle: ModelBundle) -> dict:
    """Model inputs, scores and alerts for a block of raw transactions."""
    X = build_X(df, "pre")[bundle.feature_columns]
    scores = bundle.model.predict_proba(X)[:, 1]
    return {"X": X, "scores": scores, "alerts": scores >= bundle.threshold}


def compare(ref: dict, cur: dict) -> dict:
    """Drift of one profile against the reference profile."""
    features = {}
    for col in ref["X"].columns:
        value = psi(ref["X"][col], cur["X"][col])
        features[col] = {
            "psi": value,
            "ks": ks_stat(ref["X"][col], cur["X"][col]),
            "status": status(value),
        }
    score_psi = psi(ref["scores"], cur["scores"])
    score = {
        "psi": score_psi,
        "ks": ks_stat(ref["scores"], cur["scores"]),
        "status": status(score_psi),
    }
    alerts_ref = float(ref["alerts"].mean() * 1000)
    alerts_cur = float(cur["alerts"].mean() * 1000)
    worst_col = max(features, key=lambda c: features[c]["psi"])
    overall = max(
        [score["status"], *(f["status"] for f in features.values())],
        key=STATUS_ORDER.__getitem__,
    )
    return {
        "n": len(cur["scores"]),
        "features": features,
        "score": score,
        "alerts_per_1k": alerts_cur,
        "alerts_per_1k_reference": alerts_ref,
        "worst_feature": worst_col,
        "worst_feature_psi": features[worst_col]["psi"],
        "status": overall,
    }


def apply_shift(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Artificially shift a block of transactions to test that the monitor reacts."""
    out = df.copy()
    if name == "amount_x3":
        out["amount"] = out["amount"] * 3
    elif name == "type_mix":
        out.loc[out.index[::2], "type"] = "TRANSFER"
    else:
        raise ValueError(f"Unknown shift: {name}")
    return out


SHIFTS = ("amount_x3", "type_mix")


def daily_windows(df: pd.DataFrame, min_rows: int = 1000) -> list[tuple[int, pd.DataFrame]]:
    """Split by day (step // 24), skipping days with fewer than min_rows rows."""
    day = df["step"] // 24
    return [(int(d), g) for d, g in df.groupby(day) if len(g) >= min_rows]


def _effect(base: dict, shifted: dict, top: int = 3) -> dict:
    """What an injected shift changed on the same day."""
    moves = sorted(
        (
            (col, base["features"][col]["psi"], shifted["features"][col]["psi"])
            for col in base["features"]
        ),
        key=lambda m: m[2] - m[1],
        reverse=True,
    )
    return {
        "score_psi": base["score"]["psi"],
        "score_ks": base["score"]["ks"],
        "alerts_per_1k": base["alerts_per_1k"],
        "top_feature_changes": [list(m) for m in moves[:top]],
    }


def run_monitor(
    df: pd.DataFrame,
    bundle: ModelBundle,
    shift: str | None = None,
    shift_day: int | None = None,
    min_rows: int = 1000,
) -> dict:
    """Compare each test-period day with the validation period."""
    _, val, test = time_split3(df)
    ref = profile(val, bundle)
    windows = daily_windows(test, min_rows)
    if not windows:
        raise ValueError("No test-period day has enough rows")
    target = shift_day if shift_day is not None else windows[-1][0]
    rows = []
    for day, block in windows:
        shifted = shift is not None and day == target
        extra = {}
        if shifted:
            base = compare(ref, profile(block, bundle))
            block = apply_shift(block, shift)
        row = {"day": day, "shifted": shifted, **compare(ref, profile(block, bundle))}
        if shifted:
            extra = {"unshifted": _effect(base, row)}
        rows.append({**row, **extra})
    return {
        "reference_rows": len(val),
        "threshold": bundle.threshold,
        "shift": shift,
        "shift_day": target if shift else None,
        "days": rows,
    }


def format_table(report: dict) -> str:
    lines = [
        (
            "| Day | Rows | Alerts / 1k | Reference alerts / 1k | Score PSI | Score KS "
            "| Worst feature (PSI) | Status |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in report["days"]:
        mark = " (shifted)" if d["shifted"] else ""
        lines.append(
            f"| {d['day']}{mark} | {d['n']} | {d['alerts_per_1k']:.2f} | "
            f"{d['alerts_per_1k_reference']:.2f} | {d['score']['psi']:.3f} | "
            f"{d['score']['ks']:.3f} | {d['worst_feature']} ({d['worst_feature_psi']:.3f}) | "
            f"{d['status']} |"
        )
    for d in report["days"]:
        if d["shifted"]:
            u = d["unshifted"]
            moves = ", ".join(f"{c} ({a:.3f} to {b:.3f})" for c, a, b in u["top_feature_changes"])
            lines.append(
                f"\nEffect of the injected shift on day {d['day']} (before to after): "
                f"score PSI {u['score_psi']:.3f} to {d['score']['psi']:.3f}, "
                f"score KS {u['score_ks']:.3f} to {d['score']['ks']:.3f}, "
                f"alerts per 1k {u['alerts_per_1k']:.2f} to {d['alerts_per_1k']:.2f}. "
                f"Largest feature PSI increases: {moves}."
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", MODEL_PATH))
    parser.add_argument("--simulate-shift", choices=SHIFTS, default=None)
    parser.add_argument("--shift-day", type=int, default=None)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = run_monitor(
        read_transactions(args.db),
        load_bundle(args.model),
        shift=args.simulate_shift,
        shift_day=args.shift_day,
        min_rows=args.min_rows,
    )
    suffix = f"_{args.simulate_shift}" if args.simulate_shift else ""
    out = Path(args.out or f"artifacts/drift{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_table(report))
    print("\nPSI bands: under 0.1 stable, 0.1 to 0.25 watch, over 0.25 investigate (conventions).")
    if args.simulate_shift:
        print(f"Day {report['shift_day']} was artificially shifted ({args.simulate_shift}).")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
