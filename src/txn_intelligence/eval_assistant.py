"""Evaluate the SQL assistant: 30 questions with reference SQL, scored by execution accuracy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .nl2sql import LLMFn, ask, ollama_chat, run_query, validate_sql

EVAL_SET = [
    {
        "question": "How many transactions are there?",
        "sql": "SELECT COUNT(*) FROM transactions",
    },
    {
        "question": "How many fraud transactions are there?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "What is the overall fraud rate as a percentage?",
        "sql": "SELECT 100.0 * AVG(is_fraud) FROM transactions",
    },
    {
        "question": "How many transactions are there of each type?",
        "sql": "SELECT type, COUNT(*) FROM transactions GROUP BY type",
    },
    {
        "question": "How many fraud cases are there per transaction type?",
        "sql": "SELECT type, COUNT(*) FROM transactions WHERE is_fraud = 1 GROUP BY type",
    },
    {
        "question": "What is the average amount of fraud transactions?",
        "sql": "SELECT AVG(amount) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "What is the average amount of non-fraud transactions?",
        "sql": "SELECT AVG(amount) FROM transactions WHERE is_fraud = 0",
    },
    {
        "question": "What is the largest transaction amount?",
        "sql": "SELECT MAX(amount) FROM transactions",
    },
    {
        "question": "How many TRANSFER transactions are fraud?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE type = 'TRANSFER' AND is_fraud = 1",
    },
    {
        "question": "How many CASH_OUT transactions are fraud?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE type = 'CASH_OUT' AND is_fraud = 1",
    },
    {
        "question": "How many transactions have step 24 or less?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE step <= 24",
    },
    {
        "question": "What is the highest number of fraud cases in any single step?",
        "sql": (
            "SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM transactions "
            "WHERE is_fraud = 1 GROUP BY step)"
        ),
    },
    {
        "question": "How many distinct sender accounts are there?",
        "sql": "SELECT COUNT(DISTINCT name_orig) FROM transactions",
    },
    {
        "question": "How many transactions have an amount greater than 100000?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE amount > 100000",
    },
    {
        "question": "How many fraud transactions left the sender with a balance of zero?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1 AND new_balance_orig = 0",
    },
    {
        "question": (
            "How many transactions have an amount exactly equal to the sender's old balance, "
            "where that balance is above zero?"
        ),
        "sql": (
            "SELECT COUNT(*) FROM transactions "
            "WHERE amount = old_balance_orig AND old_balance_orig > 0"
        ),
    },
    {
        "question": (
            "Among transactions with an amount exactly equal to the sender's old balance "
            "(balance above zero), how many are fraud?"
        ),
        "sql": (
            "SELECT COUNT(*) FROM transactions "
            "WHERE amount = old_balance_orig AND old_balance_orig > 0 AND is_fraud = 1"
        ),
    },
    {
        "question": "What is the average amount per transaction type?",
        "sql": "SELECT type, AVG(amount) FROM transactions GROUP BY type",
    },
    {
        "question": "What is the total amount of PAYMENT transactions?",
        "sql": "SELECT SUM(amount) FROM transactions WHERE type = 'PAYMENT'",
    },
    {
        "question": "How many transactions were flagged by the existing rule?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_flagged_fraud = 1",
    },
    {
        "question": "How many fraud transactions were not flagged by the existing rule?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1 AND is_flagged_fraud = 0",
    },
    {
        "question": "What is the fraud rate as a percentage for each transaction type?",
        "sql": "SELECT type, 100.0 * AVG(is_fraud) FROM transactions GROUP BY type",
    },
    {
        "question": "How many transactions are there per day, where day is step divided by 24?",
        "sql": (
            "SELECT step // 24 AS day, COUNT(*) FROM transactions GROUP BY day ORDER BY day"
        ),
    },
    {
        "question": "What is the median amount of fraud transactions?",
        "sql": "SELECT MEDIAN(amount) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "How many CASH_IN transactions have a receiver balance of zero before the transaction?",
        "sql": (
            "SELECT COUNT(*) FROM transactions WHERE type = 'CASH_IN' AND old_balance_dest = 0"
        ),
    },
    {
        "question": "What percentage of transactions are of type TRANSFER?",
        "sql": "SELECT 100.0 * AVG(CASE WHEN type = 'TRANSFER' THEN 1.0 ELSE 0.0 END) FROM transactions",
    },
    {
        "question": "How many distinct receiver accounts received fraud transactions?",
        "sql": "SELECT COUNT(DISTINCT name_dest) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "What is the total amount of fraud transactions?",
        "sql": "SELECT SUM(amount) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "What is the smallest amount among fraud transactions?",
        "sql": "SELECT MIN(amount) FROM transactions WHERE is_fraud = 1",
    },
    {
        "question": "How many fraud transactions happened at hour of day 0, meaning step modulo 24 equals 0?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1 AND step % 24 = 0",
    },
]


HELD_OUT_SET = [
    {
        "question": "How many transactions are of type DEBIT?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE type = 'DEBIT'",
    },
    {
        "question": "What is the total amount of CASH_OUT transactions?",
        "sql": "SELECT SUM(amount) FROM transactions WHERE type = 'CASH_OUT'",
    },
    {
        "question": "What is the average amount of fraud transactions of type TRANSFER?",
        "sql": "SELECT AVG(amount) FROM transactions WHERE is_fraud = 1 AND type = 'TRANSFER'",
    },
    {
        "question": "How many distinct receiver accounts are there?",
        "sql": "SELECT COUNT(DISTINCT name_dest) FROM transactions",
    },
    {
        "question": "What is the fraud rate as a percentage among CASH_OUT transactions?",
        "sql": "SELECT 100.0 * AVG(is_fraud) FROM transactions WHERE type = 'CASH_OUT'",
    },
    {
        "question": "How many transactions have an amount below 100?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE amount < 100",
    },
    {
        "question": "What is the highest number of transactions in any single step?",
        "sql": "SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM transactions GROUP BY step)",
    },
    {
        "question": "How many fraud transactions had a receiver balance of zero before the transaction?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1 AND old_balance_dest = 0",
    },
    {
        "question": "How many transactions are there per type among steps greater than 700?",
        "sql": "SELECT type, COUNT(*) FROM transactions WHERE step > 700 GROUP BY type",
    },
    {
        "question": "How many transactions left the sender with a balance of zero after the transaction?",
        "sql": "SELECT COUNT(*) FROM transactions WHERE new_balance_orig = 0",
    },
    {
        "question": "What is the smallest transaction amount that is greater than zero?",
        "sql": "SELECT MIN(amount) FROM transactions WHERE amount > 0",
    },
    {
        "question": "What is the average sender balance before the transaction among fraud transactions?",
        "sql": "SELECT AVG(old_balance_orig) FROM transactions WHERE is_fraud = 1",
    },
]

SETS = {"main": EVAL_SET, "held-out": HELD_OUT_SET}


def _normalise(value: object) -> object:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) or hasattr(value, "__float__"):
        return round(float(value), 2)
    return str(value)


def _rows(df: pd.DataFrame) -> list[tuple]:
    rows = (tuple(_normalise(v) for v in row) for row in df.itertuples(index=False, name=None))
    return sorted(rows, key=repr)


def results_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True if two results hold the same values, ignoring column names, row order and tiny float noise."""
    return _rows(a) == _rows(b)


def evaluate_models(
    db_path: str | Path,
    llms: dict[str, LLMFn],
    eval_set: list[dict] = EVAL_SET,
    limit: int | None = None,
) -> dict:
    items = eval_set[:limit] if limit else eval_set
    references = [run_query(db_path, validate_sql(i["sql"]), max_rows=1000) for i in items]

    results = {}
    for name, llm in llms.items():
        details = []
        for item, reference in zip(items, references, strict=True):
            start = time.perf_counter()
            out = ask(item["question"], db_path, llm, max_rows=1000)
            seconds = time.perf_counter() - start
            valid = out["error"] is None
            details.append(
                {
                    "question": item["question"],
                    "reference_sql": item["sql"],
                    "generated_sql": out["sql"],
                    "valid": valid,
                    "correct": valid and results_match(out["rows"], reference),
                    "error": out["error"],
                    "seconds": seconds,
                }
            )
        n = len(details)
        results[name] = {
            "n_questions": n,
            "valid_sql_rate": sum(d["valid"] for d in details) / n,
            "execution_accuracy": sum(d["correct"] for d in details) / n,
            "mean_seconds": sum(d["seconds"] for d in details) / n,
            "details": details,
        }
    return results


def format_table(results: dict) -> str:
    lines = [
        "| Model | Questions | Valid SQL | Execution accuracy | Mean seconds |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_questions']} | {r['valid_sql_rate'] * 100:.1f}% | "
            f"{r['execution_accuracy'] * 100:.1f}% | {r['mean_seconds']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--models", nargs="+", default=["qwen2.5-coder:3b"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", dest="which", choices=list(SETS), default="main")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    llms = {name: ollama_chat(name) for name in args.models}
    results = evaluate_models(args.db, llms, eval_set=SETS[args.which], limit=args.limit)

    suffix = "" if args.which == "main" else "_held_out"
    out = Path(args.out or f"artifacts/assistant_eval{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(format_table(results))
    for name, r in results.items():
        wrong = [d for d in r["details"] if not d["correct"]]
        print(f"\n{name}: {len(wrong)} wrong or failed")
        for d in wrong[:5]:
            print(f"  - {d['question']}\n    {d['generated_sql']!r} {d['error'] or ''}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
