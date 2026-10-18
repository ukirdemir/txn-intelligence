import pytest

from txn_intelligence.ablation import (
    ABLATIONS,
    format_table,
    permutation_importance_pr_auc,
    run_ablation,
)
from txn_intelligence.features import PRE_FEATURE_COLUMNS, build_features


def test_ablation_groups_use_valid_columns():
    for variant, dropped in ABLATIONS.items():
        assert set(dropped) <= set(PRE_FEATURE_COLUMNS), variant
        assert len(set(PRE_FEATURE_COLUMNS) - set(dropped)) > 0, variant


def test_run_ablation_structure(paysim_df):
    summary = run_ablation(paysim_df)
    n_models = 2
    assert len(summary["results"]) == len(ABLATIONS) * n_models

    for r in summary["results"]:
        assert not set(r["dropped"]) & set(r["features"])
        assert r["n_features"] == len(r["features"])

    reference = [r for r in summary["results"] if r["variant"] == "reference"]
    assert all(r["pr_auc"] > 0.8 for r in reference)
    assert "| minimal |" in format_table(summary)


def test_permutation_importance_is_sorted_and_complete(paysim_df):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = build_features(paysim_df, "pre")
    model = HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(X, y)
    imp = permutation_importance_pr_auc(model, X, y)
    assert list(imp) == sorted(imp, key=imp.get, reverse=True)
    assert set(imp) == set(PRE_FEATURE_COLUMNS)
    assert max(imp.values()) > 0


def test_permutation_requires_fraud(paysim_df):
    from sklearn.linear_model import LogisticRegression

    X, y = build_features(paysim_df, "pre")
    model = LogisticRegression(max_iter=200).fit(X, y)
    with pytest.raises(ValueError, match="no fraud"):
        permutation_importance_pr_auc(model, X, y * 0)
