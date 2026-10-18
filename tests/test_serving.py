import pytest
from fastapi.testclient import TestClient

from txn_intelligence.api import app
from txn_intelligence.features import PRE_FEATURE_COLUMNS, build_X
from txn_intelligence.serving import (
    load_bundle,
    rule_flag,
    save_bundle,
    score_transaction,
)
from txn_intelligence.train_final import fit_bundle

FRAUD_LIKE = {
    "step": 10,
    "type": "TRANSFER",
    "amount": 500.0,
    "old_balance_orig": 500.0,
    "old_balance_dest": 0.0,
}
LEGIT_LIKE = {
    "step": 10,
    "type": "PAYMENT",
    "amount": 20.0,
    "old_balance_orig": 800.0,
    "old_balance_dest": 300.0,
}


@pytest.fixture
def bundle(paysim_df):
    return fit_bundle(paysim_df)


@pytest.fixture
def client():
    app.state.bundle = None
    yield TestClient(app)
    app.state.bundle = None


def test_build_X_pre_needs_no_labels_or_post_balances(paysim_df):
    live = paysim_df.drop(columns=["is_fraud", "new_balance_orig", "new_balance_dest"])
    X = build_X(live, "pre")
    assert list(X.columns) == PRE_FEATURE_COLUMNS
    assert len(X) == len(paysim_df)


def test_bundle_roundtrip(bundle, tmp_path):
    path = save_bundle(bundle, tmp_path / "m.joblib")
    loaded = load_bundle(path)
    a = score_transaction(bundle, FRAUD_LIKE)
    b = score_transaction(loaded, FRAUD_LIKE)
    assert a == b
    assert loaded.feature_columns == PRE_FEATURE_COLUMNS


def test_fraud_like_scores_higher_than_legit_like(bundle):
    fraud = score_transaction(bundle, FRAUD_LIKE)
    legit = score_transaction(bundle, LEGIT_LIKE)
    assert fraud["fraud_score"] > legit["fraud_score"]
    assert fraud["rule_flag"] is True
    assert legit["rule_flag"] is False


def test_rule_flag_cases():
    assert rule_flag(FRAUD_LIKE)
    assert not rule_flag({**FRAUD_LIKE, "amount": 600.0})
    assert not rule_flag({**FRAUD_LIKE, "type": "PAYMENT"})
    assert not rule_flag({**FRAUD_LIKE, "amount": 0.0, "old_balance_orig": 0.0})


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": False, "model_version": None}


def test_score_without_model_returns_503(client):
    assert client.post("/score", json=FRAUD_LIKE).status_code == 503


def test_score_with_model(client, bundle):
    app.state.bundle = bundle
    assert client.get("/health").json()["model_loaded"] is True
    r = client.post("/score", json=FRAUD_LIKE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_score", "flagged", "threshold", "rule_flag", "model_version"}
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["rule_flag"] is True


@pytest.mark.parametrize(
    "patch",
    [{"type": "BITCOIN"}, {"amount": -5.0}, {"step": -1}, {"old_balance_orig": -1.0}],
)
def test_invalid_input_rejected(client, bundle, patch):
    app.state.bundle = bundle
    assert client.post("/score", json={**FRAUD_LIKE, **patch}).status_code == 422


def test_missing_field_rejected(client, bundle):
    app.state.bundle = bundle
    body = {k: v for k, v in FRAUD_LIKE.items() if k != "amount"}
    assert client.post("/score", json=body).status_code == 422
