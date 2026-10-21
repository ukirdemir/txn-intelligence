import numpy as np
import pytest

from txn_intelligence.monitor import (
    apply_shift,
    daily_windows,
    format_table,
    ks_stat,
    psi,
    run_monitor,
    status,
)
from txn_intelligence.train_final import fit_bundle


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_identical_is_zero(rng):
    x = rng.normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_same_distribution_is_small(rng):
    assert psi(rng.normal(size=20000), rng.normal(size=20000)) < 0.02


def test_psi_detects_shift(rng):
    assert psi(rng.normal(size=5000), rng.normal(loc=1.0, size=5000)) > 0.25


def test_psi_binary_column():
    ref = np.array([0] * 500 + [1] * 500)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, np.array([0] * 900 + [1] * 100)) > 0.25


def test_psi_unseen_category_is_large():
    ref = np.array([0, 1] * 500)
    assert psi(ref, np.array([2] * 1000)) > 0.25


def test_psi_empty_raises():
    with pytest.raises(ValueError):
        psi([], [1.0])


def test_ks_identical_and_shifted(rng):
    x = rng.normal(size=3000)
    assert ks_stat(x, x) == 0.0
    assert ks_stat(x, rng.normal(loc=2.0, size=3000)) > 0.5


def test_status_bands():
    assert status(0.05) == "stable"
    assert status(0.1) == "watch"
    assert status(0.2499) == "watch"
    assert status(0.25) == "investigate"


def test_apply_shift(paysim_df):
    tripled = apply_shift(paysim_df, "amount_x3")
    assert np.allclose(tripled["amount"], paysim_df["amount"] * 3)
    mixed = apply_shift(paysim_df, "type_mix")
    assert (mixed["type"] == "TRANSFER").mean() > 0.5
    assert (paysim_df["type"] == "TRANSFER").mean() < 0.5
    with pytest.raises(ValueError):
        apply_shift(paysim_df, "nope")


def test_daily_windows_respects_min_rows(paysim_df):
    assert daily_windows(paysim_df, min_rows=10**9) == []
    windows = daily_windows(paysim_df, min_rows=1)
    assert sum(len(g) for _, g in windows) == len(paysim_df)


def test_run_monitor_flags_injected_shift(paysim_df):
    bundle = fit_bundle(paysim_df)
    clean = run_monitor(paysim_df, bundle, min_rows=50)
    shifted = run_monitor(paysim_df, bundle, shift="amount_x3", min_rows=50)
    assert len(clean["days"]) >= 2
    assert not any(d["shifted"] for d in clean["days"])
    hit = [d for d in shifted["days"] if d["shifted"]]
    others = [d for d in shifted["days"] if not d["shifted"]]
    assert len(hit) == 1
    assert hit[0]["status"] == "investigate"
    assert hit[0]["worst_feature_psi"] > max(d["worst_feature_psi"] for d in others)
    assert "(shifted)" in format_table(shifted)
    effect = hit[0]["unshifted"]
    assert hit[0]["score"]["psi"] > effect["score_psi"]
    assert "log_amount" in [c for c, _, _ in effect["top_feature_changes"]]
    assert "Effect of the injected shift" in format_table(shifted)


def test_run_monitor_no_usable_days_raises(paysim_df):
    bundle = fit_bundle(paysim_df)
    with pytest.raises(ValueError):
        run_monitor(paysim_df, bundle, min_rows=10**9)
