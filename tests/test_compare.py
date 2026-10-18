import numpy as np
import pytest

from txn_intelligence.evaluate import precision_at_recall
from txn_intelligence.features import build_features
from txn_intelligence.models import TorchMLP
from txn_intelligence.train_compare import compare_models, format_table


def test_precision_at_recall_hand_computed():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_recall(y, s, target_recall=1.0) == pytest.approx(2 / 3)


def test_precision_at_recall_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert precision_at_recall(y, s, target_recall=0.9) == pytest.approx(1.0)


def test_torch_mlp_learns_separable_data(paysim_df):
    from sklearn.metrics import average_precision_score

    X, y = build_features(paysim_df)
    model = TorchMLP(epochs=10, batch_size=64, seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert average_precision_score(y, proba[:, 1]) > 0.9


def test_torch_mlp_is_deterministic(paysim_df):
    X, y = build_features(paysim_df)
    p1 = TorchMLP(epochs=1, batch_size=256, seed=3).fit(X, y).predict_proba(X)
    p2 = TorchMLP(epochs=1, batch_size=256, seed=3).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)


def test_pre_features_ignore_post_transaction_columns(paysim_df):
    import pandas as pd

    from txn_intelligence.features import PRE_FEATURE_COLUMNS

    X, _ = build_features(paysim_df, "pre")
    assert list(X.columns) == PRE_FEATURE_COLUMNS

    changed = paysim_df.copy()
    changed["new_balance_orig"] = 999.0
    changed["new_balance_dest"] = 999.0
    X2, _ = build_features(changed, "pre")
    pd.testing.assert_frame_equal(X, X2)


def test_rule_baseline_flags_drained_balances(paysim_df):
    from txn_intelligence.models import RuleBaseline

    X, y = build_features(paysim_df, "pre")
    for mode in ("exceeds", "equals"):
        scores = RuleBaseline(mode).fit(X, y).predict_proba(X)[:, 1]
        assert np.array_equal(scores, y.to_numpy().astype(float)), mode


def test_rule_equals_is_stricter_than_exceeds():
    import pandas as pd

    from txn_intelligence.models import RuleBaseline

    X = pd.DataFrame(
        {
            "type_TRANSFER": [1, 1, 1],
            "type_CASH_OUT": [0, 0, 0],
            "orig_would_be_emptied": [1, 1, 0],
            "amount_to_orig_balance": [1.0, 2.5, 0.4],
        }
    )
    assert RuleBaseline("exceeds").predict_proba(X)[:, 1].tolist() == [1.0, 1.0, 0.0]
    assert RuleBaseline("equals").predict_proba(X)[:, 1].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="mode"):
        RuleBaseline("bogus")


def test_rule_baseline_requires_pre_features(paysim_df):
    from txn_intelligence.models import RuleBaseline

    X_full, y = build_features(paysim_df, "full")
    with pytest.raises(ValueError, match="pre"):
        RuleBaseline().fit(X_full, y)


def test_unknown_feature_set_rejected(paysim_df):
    with pytest.raises(ValueError, match="feature_set"):
        build_features(paysim_df, "bogus")


def test_compare_models_pre_features(paysim_df):
    summary = compare_models(paysim_df, epochs=10, batch_size=64, feature_set="pre")
    assert summary["feature_set"] == "pre"
    assert [r["model"] for r in summary["results"][:2]] == [
        "rule_balance_exceeded",
        "rule_balance_equal",
    ]
    by_name = {r["model"]: r for r in summary["results"]}
    assert by_name["rule_balance_exceeded"]["pr_auc"] > 0.99
    assert by_name["rule_balance_equal"]["pr_auc"] > 0.99
    assert by_name["logistic_regression"]["pr_auc"] > 0.8
    assert by_name["hist_gradient_boosting"]["pr_auc"] > 0.8


def test_compare_models_summary(paysim_df):
    summary = compare_models(paysim_df, epochs=10, batch_size=64)
    names = [r["model"] for r in summary["results"]]
    assert names == ["logistic_regression", "hist_gradient_boosting", "torch_mlp"]
    for r in summary["results"]:
        assert r["n_missed_fraud"] == sum(r["missed_fraud_by_type"].values())
        if r["model"] == "torch_mlp":
            assert r["pr_auc"] > 0.5
        else:
            assert r["pr_auc"] > 0.8
            assert r["cost_per_1k"] < summary["cost_per_1k_flag_nothing"]
    assert summary["missed_by_all_models"] <= min(r["n_missed_fraud"] for r in summary["results"])
    assert "| torch_mlp |" in format_table(summary)
