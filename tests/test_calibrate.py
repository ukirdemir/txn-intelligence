import numpy as np
import pytest

from txn_intelligence.calibrate import (
    calibration_metrics,
    fit_isotonic,
    fit_platt,
    format_table,
    period_groups,
    reliability,
    run_calibration,
    theory_threshold,
)
from txn_intelligence.config import COSTS


@pytest.fixture
def miscalibrated():
    rng = np.random.default_rng(0)
    n = 40000
    p_true = rng.beta(0.6, 4, n)
    y = rng.random(n) < p_true
    return y, p_true**3, p_true


def test_theory_threshold_matches_cost_matrix():
    expected = COSTS.false_positive / (
        COSTS.false_positive + COSTS.false_negative - COSTS.true_positive
    )
    assert theory_threshold() == pytest.approx(expected)
    assert 0.0 < theory_threshold() < 0.1


def test_isotonic_is_monotone_and_bounded(miscalibrated):
    y, s, _ = miscalibrated
    mapper = fit_isotonic(y, s)
    grid = np.linspace(0, 1, 500)
    out = mapper(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_platt_is_increasing(miscalibrated):
    y, s, _ = miscalibrated
    out = fit_platt(y, s)(np.linspace(0.001, 0.999, 200))
    assert np.all(np.diff(out) > 0)


def test_calibration_improves_brier(miscalibrated):
    y, s, _ = miscalibrated
    raw = calibration_metrics(y, s)["brier"]
    assert calibration_metrics(y, fit_isotonic(y, s)(s))["brier"] < raw
    assert calibration_metrics(y, fit_platt(y, s)(s))["brier"] < raw


def test_ece_small_for_calibrated_probabilities(miscalibrated):
    y, _, p_true = miscalibrated
    assert calibration_metrics(y, p_true)["ece"] < 0.02


def test_reliability_counts_sum_to_n(miscalibrated):
    y, s, _ = miscalibrated
    assert sum(b["n"] for b in reliability(y, s)) == len(y)


def test_period_groups_split_by_day_volume():
    steps = np.array([1] * 5 + [30] * 2 + [60] * 5)
    groups = period_groups(steps, volume_split=5)
    assert groups["high_volume_days"].sum() == 10
    assert groups["thin_days"].sum() == 2
    assert list(period_groups(steps, volume_split=100)) == ["thin_days"]


def test_run_calibration_structure(paysim_df):
    out = run_calibration(paysim_df, volume_split=150)
    assert set(out["methods"]) == {"raw_score", "isotonic", "platt"}
    assert "isotonic_at_theory_threshold" in out["decisions"]
    assert out["periods"]
    assert sum(p["n"] for p in out["periods"].values()) == out["n_test"]
    table = format_table(out)
    assert "isotonic" in table
    assert "platt" in table
