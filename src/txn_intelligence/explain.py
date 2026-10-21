"""Template explanations for a score, built only from facts checked in code."""

from __future__ import annotations

FRAUD_TYPES = ("TRANSFER", "CASH_OUT")

NOTE = (
    "These are signals present in the input, not a per-transaction attribution of the model. "
    "In PaySim training data the account-emptying pattern carried most of the score."
)


def _money(value: float) -> str:
    return f"{value:,.2f}"


def reasons(txn: dict, result: dict) -> list[str]:
    """Plain-language facts about the transaction and the decision."""
    out = []
    side = "at or above" if result["flagged"] else "below"
    out.append(
        f"Score {result['fraud_score']:.3f} is {side} the flag threshold {result['threshold']:.3f}."
    )

    if txn["type"] in FRAUD_TYPES:
        out.append(f"Type is {txn['type']}, one of the two types where fraud occurs in the data.")
    else:
        out.append(f"Type is {txn['type']}; no fraud occurs in this type in the data.")

    amount, balance = txn["amount"], txn["old_balance_orig"]
    if balance == 0:
        out.append("Sender balance before the transaction is 0.")
    elif amount == balance:
        out.append(
            f"Amount {_money(amount)} equals the sender's balance {_money(balance)} "
            "(the account is emptied)."
        )
    elif amount > balance:
        out.append(
            f"Amount {_money(amount)} exceeds the sender's balance {_money(balance)} "
            "(common in legitimate PaySim rows too)."
        )
    else:
        out.append(f"Amount {_money(amount)} is below the sender's balance {_money(balance)}.")

    if txn["old_balance_dest"] == 0:
        out.append("Receiver balance before the transaction is 0.")

    if result["rule_flag"] == result["flagged"]:
        out.append("The one-line rule (amount equals balance) agrees with the decision.")
    else:
        verb = "flags" if result["rule_flag"] else "does not flag"
        out.append(f"The one-line rule (amount equals balance) disagrees: it {verb} this one.")
    return out


def explain(txn: dict, result: dict) -> dict:
    """Return the decision summary, reasons and the standing caveat."""
    return {
        "reasons": reasons(txn, result),
        "note": NOTE,
    }
