import pytest
from fastapi.testclient import TestClient

from txn_intelligence.api import app
from txn_intelligence.explain import NOTE, explain, reasons
from txn_intelligence.train_final import fit_bundle

DRAIN = {
    "step": 10,
    "type": "TRANSFER",
    "amount": 181.0,
    "old_balance_orig": 181.0,
    "old_balance_dest": 0.0,
}
SMALL = {
    "step": 10,
    "type": "PAYMENT",
    "amount": 12.5,
    "old_balance_orig": 4000.0,
    "old_balance_dest": 300.0,
}


def result(flagged: bool, rule: bool, score: float = 0.9):
    return {"fraud_score": score, "flagged": flagged, "threshold": 0.5, "rule_flag": rule}


def test_draining_transfer_reasons():
    text = " ".join(reasons(DRAIN, result(True, True)))
    assert "at or above" in text
    assert "equals the sender's balance" in text
    assert "Receiver balance before the transaction is 0" in text
    assert "agrees" in text


def test_small_payment_reasons():
    text = " ".join(reasons(SMALL, result(False, False, 0.01)))
    assert "below the flag threshold" in text
    assert "no fraud occurs in this type" in text
    assert "is below the sender's balance" in text
    assert "Receiver balance" not in text


def test_exceeds_balance_wording():
    txn = {**DRAIN, "amount": 500.0}
    text = " ".join(reasons(txn, result(False, False)))
    assert "exceeds the sender's balance" in text


def test_zero_sender_balance():
    txn = {**DRAIN, "amount": 0.0, "old_balance_orig": 0.0}
    assert "Sender balance before the transaction is 0." in reasons(txn, result(False, False))


def test_rule_disagreement_both_ways():
    flagged_no_rule = " ".join(reasons(SMALL, result(True, False)))
    rule_not_flagged = " ".join(reasons(DRAIN, result(False, True)))
    assert "does not flag" in flagged_no_rule
    assert "it flags this one" in rule_not_flagged


def test_explain_has_note():
    out = explain(DRAIN, result(True, True))
    assert out["note"] == NOTE
    assert out["reasons"]


@pytest.fixture
def client(paysim_df):
    app.state.bundle = fit_bundle(paysim_df)
    yield TestClient(app)
    app.state.bundle = None


def test_api_explain_off_by_default(client):
    body = client.post("/score", json=DRAIN).json()
    assert "reasons" not in body
    assert "note" not in body


def test_api_explain_on(client):
    body = client.post("/score?explain=true", json=DRAIN).json()
    assert body["reasons"]
    assert body["note"] == NOTE
    assert body["rule_flag"] is True
