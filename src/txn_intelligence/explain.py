"""Explanations for a score: template reasons from code, optionally rephrased by a local LLM."""

from __future__ import annotations

import argparse
import os
import re

from .answer import NUMBER_WORDS, numbers_in
from .config import MODEL_PATH
from .nl2sql import LLMFn, LLMUnavailable, ollama_chat
from .serving import load_bundle, score_transaction

FRAUD_TYPES = ("TRANSFER", "CASH_OUT")

NOTE = (
    "These are signals present in the input, not a per-transaction attribution of the model. "
    "In PaySim training data the account-emptying pattern carried most of the score."
)

REPHRASE_PROMPT = """Rewrite the facts below as a short explanation of 2 or 3 sentences.

Rules:
- Use only the facts given. Do not add numbers, causes, advice or opinions.
- The facts are signals, not causes. Never join them with "because", "as", "since", "due to",
  "so" or "which is why". Join them with "and" or "while", or write separate short sentences.
- Write numbers as digits, exactly as given.
- Start with "Flagged." or "Not flagged." and then state the facts.
"""

CAUSAL = re.compile(
    r"\b(because|due to|caused|causes|led to|leads to|as a result|therefore|thus|hence"
    r"|as it|as the|which is why|resulting in|owing to|thanks to)\b",
    re.IGNORECASE,
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


def _decision_consistent(text: str, flagged: bool) -> bool:
    lowered = text.lower()
    negative = "not flagged" in lowered or "below" in lowered
    positive = "at or above" in lowered or ("flagged" in lowered and "not flagged" not in lowered)
    return (positive and not negative) if flagged else (negative and not positive)


def problems(text: str, facts: list[str], flagged: bool) -> list[str]:
    """Reasons a rephrased text must be rejected; empty when it is faithful."""
    found = []
    if not text.strip():
        return ["empty reply"]
    source = " ".join(facts)
    allowed = {value for value, _ in numbers_in(source)}
    extra = [value for value, _ in numbers_in(text) if value not in allowed]
    if extra:
        found.append(f"numbers not in the facts: {extra}")
    allowed_words = {w.lower() for w in NUMBER_WORDS.findall(source)}
    words = [w for w in NUMBER_WORDS.findall(text) if w.lower() not in allowed_words]
    if words:
        found.append(f"number words not in the facts: {words}")
    if CAUSAL.search(text):
        found.append("causal wording")
    if not _decision_consistent(text, flagged):
        found.append("decision wording does not match")
    return found


def rephrase_messages(facts: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": REPHRASE_PROMPT},
        {"role": "user", "content": "Facts:\n" + "\n".join(f"- {f}" for f in facts)},
    ]


def rephrase(
    facts: list[str], flagged: bool, llm: LLMFn, max_retries: int = 1
) -> tuple[str, str]:
    """Return (text, source); "llm" if a rewrite passed every check, else the template text."""
    fallback = " ".join(facts)
    messages = rephrase_messages(facts)
    for _ in range(max_retries + 1):
        try:
            text = llm(messages).strip()
        except (LLMUnavailable, OSError):
            return fallback, "template"
        found = problems(text, facts, flagged)
        if not found:
            return text, "llm"
        messages = [
            *messages,
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "Rejected: " + "; ".join(found) + ". Rewrite it using only the facts.",
            },
        ]
    return fallback, "template"


def main(argv: list[str] | None = None) -> None:
    """Score one transaction and show the model's raw rewrite and why it passed or failed."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--bundle", default=os.environ.get("MODEL_PATH", MODEL_PATH))
    parser.add_argument("--model", default=os.environ.get("EXPLAIN_MODEL", "qwen2.5-coder:7b"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--type", default="TRANSFER")
    parser.add_argument("--amount", type=float, default=181.0)
    parser.add_argument("--old-balance-orig", type=float, default=181.0)
    parser.add_argument("--old-balance-dest", type=float, default=0.0)
    args = parser.parse_args(argv)

    txn = {
        "step": args.step,
        "type": args.type,
        "amount": args.amount,
        "old_balance_orig": args.old_balance_orig,
        "old_balance_dest": args.old_balance_dest,
    }
    result = score_transaction(load_bundle(args.bundle), txn)
    facts = reasons(txn, result)
    print("Facts:")
    for fact in facts:
        print(f"- {fact}")
    try:
        text = ollama_chat(args.model, timeout=args.timeout)(rephrase_messages(facts)).strip()
    except (LLMUnavailable, OSError) as exc:
        raise SystemExit(f"\nNo model reply ({type(exc).__name__}): {exc}") from exc
    print(f"\nModel reply:\n{text}")
    print(f"\nProblems: {problems(text, facts, result['flagged']) or 'none'}")


def explain(txn: dict, result: dict, llm: LLMFn | None = None) -> dict:
    """Return the reasons and the standing caveat, plus an LLM-phrased text when llm is given."""
    facts = reasons(txn, result)
    out = {"reasons": facts, "note": NOTE}
    if llm is not None:
        out["explanation"], out["explanation_source"] = rephrase(facts, result["flagged"], llm)
    return out


if __name__ == "__main__":
    main()
