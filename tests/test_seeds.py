import pytest

from txn_intelligence.seeds import aggregate, format_table, make_model, run_seeds


def test_aggregate_values():
    out = aggregate([1.0, 2.0, 3.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["std"] == pytest.approx(1.0)
    assert out["min"] == 1.0
    assert out["max"] == 3.0
    assert aggregate([5.0])["std"] == 0.0


def test_make_model_unknown_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("bogus", 0)


def test_n_seeds_validated(paysim_df):
    with pytest.raises(ValueError, match="n_seeds"):
        run_seeds(paysim_df, n_seeds=0)


def test_gbm_runs_structure_and_bounds(paysim_df):
    out = run_seeds(paysim_df, ("hist_gradient_boosting",), n_seeds=2)
    runs = out["models"]["hist_gradient_boosting"]["runs"]
    assert [r["seed"] for r in runs] == [0, 1]
    for r in runs:
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.0 <= r["pr_auc"] <= 1.0
        assert r["alerts_per_1k"] >= 0.0
    assert out["models"]["hist_gradient_boosting"]["cost_per_1k"]["min"] <= (
        out["models"]["hist_gradient_boosting"]["cost_per_1k"]["max"]
    )


def test_mlp_runs_and_table(paysim_df):
    out = run_seeds(paysim_df, ("torch_mlp",), n_seeds=2, epochs=2, batch_size=64)
    assert len(out["models"]["torch_mlp"]["runs"]) == 2
    table = format_table(out)
    assert "| torch_mlp |" in table
    assert "±" in table
