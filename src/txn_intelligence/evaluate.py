"""Cost-aware metrics and threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from .config import COSTS, CostMatrix


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    costs: CostMatrix = COSTS,
) -> float:
    """Total cost when flagging every transaction with score >= threshold."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores) >= threshold

    fn = np.sum(y_true & ~flagged)
    fp = np.sum(~y_true & flagged)
    tp = np.sum(y_true & flagged)
    return float(fn * costs.false_negative + fp * costs.false_positive + tp * costs.true_positive)


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    costs: CostMatrix = COSTS,
) -> tuple[float, float]:
    """Return (threshold, cost) minimising expected cost, in O(n log n)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true).astype(int)
    n = len(scores)
    positives = int(y.sum())

    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    fp = k - tp
    fn = positives - tp
    cost = fn * costs.false_negative + fp * costs.false_positive + tp * costs.true_positive

    cut_idx = np.r_[np.flatnonzero(s[1:] != s[:-1]), n - 1]
    best = int(cut_idx[np.argmin(cost[cut_idx])])
    best_cost = float(cost[best])
    best_t = float(s[best])

    flag_nothing_cost = float(positives * costs.false_negative)
    if flag_nothing_cost < best_cost:
        return float(s[0]) + 1e-9, flag_nothing_cost
    return best_t, best_cost


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, scores))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Recall (true-positive rate) at a fixed false-positive rate."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def precision_at_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.9
) -> float:
    """Best precision while catching at least `target_recall` of fraud."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    mask = recall >= target_recall
    return float(precision[mask].max()) if mask.any() else 0.0


def cost_per_1k(total_cost: float, n: int) -> float:
    """Normalise a total cost to 'per 1,000 transactions'."""
    return total_cost / n * 1000.0
