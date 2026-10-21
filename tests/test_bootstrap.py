import numpy as np
import pytest

from txn_intelligence.bootstrap import (
    bootstrap,
    format_table,
    poisson_weights,
    prepare,
    run_bootstrap,
    summarise,
    weighted_metrics,
)
from txn_intelligence.config import COSTS
from txn_intelligence.evaluate import cost_per_1k, expected_cost, pr_auc


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    n = 6000
    y = rng.random(n) < 0.05
    good = np.where(y, rng.beta(5, 2, n), rng.beta(2, 5, n))
    weak = rng.random(n)
    return y, {"good": good, "weak": weak}, {"good": 0.5, "weak": 0.9}


def test_unit_weights_match_reference_metrics(data):
    y, scores, thresholds = data
    m = weighted_metrics(prepare(y, scores["good"], thresholds["good"]), np.ones(len(y)))
    total = expected_cost(y, scores["good"], thresholds["good"], COSTS)
    assert m["cost_per_1k"] == pytest.approx(cost_per_1k(total, len(y)))
    assert m["pr_auc"] == pytest.approx(pr_auc(y, scores["good"]), abs=1e-9)
    flagged = scores["good"] >= thresholds["good"]
    assert m["recall"] == pytest.approx((y & flagged).sum() / y.sum())
    assert m["alerts_per_1k"] == pytest.approx(flagged.mean() * 1000)


def test_pr_auc_handles_tied_scores(data):
    y, _, _ = data
    ties = (np.random.default_rng(2).random(len(y)) > 0.5).astype(float)
    m = weighted_metrics(prepare(y, ties, 0.5), np.ones(len(y)))
    assert m["pr_auc"] == pytest.approx(pr_auc(y, ties), abs=1e-9)


def test_poisson_weights_rows_and_groups():
    rng = np.random.default_rng(0)
    assert poisson_weights(100000, rng).mean() == pytest.approx(1.0, abs=0.02)
    groups = np.repeat([0, 1, 2], 50)
    w = poisson_weights(150, rng, groups)
    for g in range(3):
        assert len(set(w[groups == g])) == 1


def test_bootstrap_reproducible_and_seed_sensitive(data):
    y, scores, thresholds = data
    a = bootstrap(y, scores, thresholds, n_boot=30, seed=0)
    b = bootstrap(y, scores, thresholds, n_boot=30, seed=0)
    c = bootstrap(y, scores, thresholds, n_boot=30, seed=1)
    assert np.array_equal(a["draws"]["good"]["cost_per_1k"], b["draws"]["good"]["cost_per_1k"])
    assert not np.array_equal(a["draws"]["good"]["cost_per_1k"], c["draws"]["good"]["cost_per_1k"])


def test_intervals_contain_point_estimates(data):
    y, scores, thresholds = data
    out = summarise(bootstrap(y, scores, thresholds, n_boot=200), thresholds, "good")
    for model in out["models"].values():
        for metric in ("cost_per_1k", "recall", "pr_auc", "alerts_per_1k"):
            cell = model[metric]
            assert cell["lo"] <= cell["point"] <= cell["hi"]


def test_paired_difference_and_identical_model(data):
    y, scores, thresholds = data
    scores = {**scores, "copy": scores["good"].copy()}
    thresholds = {**thresholds, "copy": thresholds["good"]}
    out = summarise(bootstrap(y, scores, thresholds, n_boot=100), thresholds, "good")
    same = out["paired"]["copy"]
    assert same["cost_other_minus_reference"]["lo"] == 0.0
    assert same["cost_other_minus_reference"]["hi"] == 0.0
    assert same["share_reference_cheaper"] == 0.0
    assert out["paired"]["weak"]["share_reference_cheaper"] > 0.99


def test_run_bootstrap_structure(paysim_df):
    models = ("rule_balance_equal", "logistic_regression")
    out = run_bootstrap(paysim_df, models, n_boot=20, reference="logistic_regression")
    assert set(out["models"]) == set(models)
    assert out["n_boot"] == 20
    assert "rule_balance_equal" in out["paired"]
    assert "| logistic_regression |" in format_table(out)


def test_run_bootstrap_by_days_runs(paysim_df):
    out = run_bootstrap(
        paysim_df, ("rule_balance_equal",), method="days", n_boot=10, reference="none"
    )
    assert out["method"] == "days"
    assert out["paired"] == {}


def test_unknown_method_rejected(paysim_df):
    with pytest.raises(ValueError, match="Unknown method"):
        run_bootstrap(paysim_df, ("rule_balance_equal",), method="bogus")
