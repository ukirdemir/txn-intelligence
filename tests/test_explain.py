import pytest
from fastapi.testclient import TestClient

from txn_intelligence.api import app
from txn_intelligence.explain import NOTE, explain, problems, reasons
from txn_intelligence.nl2sql import LLMUnavailable
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


FAITHFUL = (
    "The transaction was flagged: the score 0.998 is at or above the threshold 0.998. "
    "It is a TRANSFER whose amount 181.00 equals the sender's balance 181.00, "
    "and the one-line rule agrees."
)


def fixed(text: str):
    return lambda messages: text


def drain_facts():
    return reasons(DRAIN, result(True, True, 0.998) | {"threshold": 0.998})


def test_faithful_rewrite_accepted():
    out = explain(DRAIN, result(True, True, 0.998) | {"threshold": 0.998}, fixed(FAITHFUL))
    assert out["explanation_source"] == "llm"
    assert out["explanation"] == FAITHFUL


@pytest.mark.parametrize(
    "text",
    [
        FAITHFUL + " It is 99 percent likely fraud.",
        FAITHFUL.replace("181.00 equals", "180.00 equals"),
        FAITHFUL + " About twenty customers were affected.",
        FAITHFUL + " It was flagged because the account was emptied.",
        FAITHFUL + " It was flagged as it met the threshold.",
        FAITHFUL + " Hence the rule confirmed it.",
        "The transaction was not flagged: the score 0.998 is below the threshold 0.998.",
        "   ",
    ],
)
def test_unfaithful_rewrite_falls_back_to_template(text):
    res = result(True, True, 0.998) | {"threshold": 0.998}
    out = explain(DRAIN, res, fixed(text))
    assert out["explanation_source"] == "template"
    assert out["explanation"] == " ".join(out["reasons"])


def test_retry_recovers_after_rejection():
    replies = iter([FAITHFUL + " It was flagged as it met the threshold.", FAITHFUL])
    seen = []

    def llm(messages):
        seen.append(messages)
        return next(replies)

    out = explain(DRAIN, result(True, True, 0.998) | {"threshold": 0.998}, llm)
    assert out["explanation_source"] == "llm"
    assert out["explanation"] == FAITHFUL
    assert len(seen) == 2
    assert "Rejected" in seen[1][-1]["content"]
    assert "causal" in seen[1][-1]["content"]


def test_gives_up_after_one_retry():
    calls = []

    def llm(messages):
        calls.append(1)
        return FAITHFUL + " It was flagged because the account was emptied."

    out = explain(DRAIN, result(True, True, 0.998) | {"threshold": 0.998}, llm)
    assert out["explanation_source"] == "template"
    assert len(calls) == 2


def test_llm_unavailable_falls_back():
    def down(messages):
        raise LLMUnavailable("no ollama")

    out = explain(DRAIN, result(True, True), down)
    assert out["explanation_source"] == "template"


def test_no_llm_means_no_explanation_fields():
    out = explain(DRAIN, result(True, True))
    assert "explanation" not in out
    assert "explanation_source" not in out


def test_problems_empty_for_faithful_text():
    facts = drain_facts()
    assert problems(FAITHFUL, facts, True) == []


@pytest.fixture
def client(paysim_df):
    app.state.bundle = fit_bundle(paysim_df)
    yield TestClient(app)
    app.state.bundle = None


def test_api_explain_off_by_default(client):
    body = client.post("/score", json=DRAIN).json()
    assert "reasons" not in body
    assert "note" not in body


def test_api_llm_explanation_and_fallback(client, monkeypatch):
    def good(messages):
        return "The transaction was flagged; the score and threshold are both high."

    monkeypatch.setattr("txn_intelligence.api._explain_llm", lambda: good)
    body = client.post("/score?explain=true&llm=true", json=DRAIN).json()
    assert body["explanation_source"] in {"llm", "template"}
    assert body["explanation"]

    def down(messages):
        raise LLMUnavailable("no ollama")

    monkeypatch.setattr("txn_intelligence.api._explain_llm", lambda: down)
    body = client.post("/score?explain=true&llm=true", json=DRAIN).json()
    assert body["explanation_source"] == "template"
    assert body["explanation"] == " ".join(body["reasons"])


def test_api_llm_ignored_without_explain(client):
    body = client.post("/score?llm=true", json=DRAIN).json()
    assert "explanation" not in body


def test_api_explain_on(client):
    body = client.post("/score?explain=true", json=DRAIN).json()
    assert body["reasons"]
    assert body["note"] == NOTE
    assert body["rule_flag"] is True
