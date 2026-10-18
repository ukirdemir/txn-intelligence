import pytest

from txn_intelligence.sensitivity import format_table, run_sensitivity

MODELS = ("rule_balance_equal", "logistic_regression")
FN_COSTS = (10, 100, 500)


@pytest.fixture
def summary(paysim_df):
    return run_sensitivity(paysim_df, model_names=MODELS, fn_costs=FN_COSTS)


def test_summary_structure(summary):
    assert len(summary["rows"]) == len(MODELS) * len(FN_COSTS)
    for r in summary["rows"]:
        assert r["ratio"] == pytest.approx(r["fn_cost"] / summary["false_positive_cost"])
        assert r["model"] in MODELS
        assert 0 <= r["n_missed_fraud"] <= summary["n_test_fraud"]
    assert set(summary["best_model_by_fn_cost"]) == {str(c) for c in FN_COSTS}


def test_higher_miss_cost_never_flags_fewer(summary):
    for model in MODELS:
        alerts = {
            r["fn_cost"]: r["alerts_per_1k"] for r in summary["rows"] if r["model"] == model
        }
        assert alerts[10.0] <= alerts[100.0] <= alerts[500.0]


def test_good_model_beats_flag_nothing(summary):
    rule_rows = [r for r in summary["rows"] if r["model"] == "rule_balance_equal"]
    assert all(r["reduction_vs_nothing_pct"] > 50 for r in rule_rows)


def test_unknown_model_rejected(paysim_df):
    with pytest.raises(ValueError, match="Unknown models"):
        run_sensitivity(paysim_df, model_names=("bogus",), fn_costs=(100,))


def test_format_table_has_every_row(summary):
    table = format_table(summary)
    assert table.count("\n") == len(summary["rows"]) + 1
    assert "| rule_balance_equal |" in table
