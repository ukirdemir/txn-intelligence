"""Natural-language questions to safe, read-only DuckDB queries via a local LLM."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd

LLMFn = Callable[[list[dict]], str]

SYSTEM_PROMPT = """You translate questions about a transactions table into one DuckDB SQL query.

Table `transactions` (PaySim, synthetic mobile-money data), one row per transaction:
- step (integer): hour index from 1 to 743; day = step // 24, hour of day = step % 24
- type (text): CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER
- amount (double): transaction amount
- name_orig (text): sender account id
- old_balance_orig, new_balance_orig (double): sender balance before and after
- name_dest (text): receiver account id
- old_balance_dest, new_balance_dest (double): receiver balance before and after
- is_fraud (integer): 1 for fraud, 0 otherwise
- is_flagged_fraud (integer): 1 if an existing rule flagged it, 0 otherwise

Rules:
- Answer with exactly one SELECT statement and nothing else.
- Use only the table `transactions`.
- If the request asks to change data, or is not a question about this table, answer exactly: REFUSE
- Use // for integer division and % for modulo.
- For percentages use expressions like 100.0 * AVG(is_fraud).
- "After the transaction" means new_balance_orig or new_balance_dest; "before" means old_balance_*.
- For "highest number of X in any single step", count per step in a subquery, then take MAX of the counts.

Examples:
Question: How many CASH_IN transactions are there?
SELECT COUNT(*) FROM transactions WHERE type = 'CASH_IN'

Question: What is the average amount of TRANSFER transactions?
SELECT AVG(amount) FROM transactions WHERE type = 'TRANSFER'

Question: What is the highest number of PAYMENT transactions in any single step?
SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM transactions WHERE type = 'PAYMENT' GROUP BY step)
"""

FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|export|import|install|load|pragma|call|set|create|drop|insert"
    r"|update|delete|alter|truncate|replace|read_csv|read_csv_auto|read_parquet|read_json"
    r"|read_text|read_blob|glob|sniff_csv|parquet_scan|csv_scan)\b",
    re.IGNORECASE,
)


REFUSAL = "This assistant is read-only and only answers questions about the transactions table."


class UnsafeSQL(ValueError):
    """Raised when generated SQL is not a single, read-only SELECT."""


class LLMUnavailable(RuntimeError):
    """Raised when the local LLM server cannot be reached."""


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply, dropping code fences and leading prose."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else text).strip()
    if not re.match(r"(select|with)\b", sql, re.IGNORECASE):
        found = re.search(r"\bselect\b", sql, re.IGNORECASE)
        if found:
            sql = sql[found.start() :]
    return sql.strip()


def validate_sql(sql: str) -> str:
    """Return the cleaned statement, or raise UnsafeSQL."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQL("Empty query")
    if ";" in cleaned:
        raise UnsafeSQL("Only a single statement is allowed")
    if "--" in cleaned or "/*" in cleaned:
        raise UnsafeSQL("Comments are not allowed")
    if not re.match(r"(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeSQL("Only SELECT queries are allowed")
    blocked = FORBIDDEN.search(cleaned)
    if blocked:
        raise UnsafeSQL(f"Forbidden keyword: {blocked.group(1).lower()}")
    return cleaned


def run_query(
    db_path: str | Path, sql: str, max_rows: int = 100, timeout_s: float = 15.0
) -> pd.DataFrame:
    """Run SQL on a read-only connection with external access off, a row cap and a timeout."""
    con = duckdb.connect(str(db_path), read_only=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    try:
        con.execute("SET enable_external_access = false")
        timer.start()
        return con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {int(max_rows)}").df()
    finally:
        timer.cancel()
        con.close()


def ollama_chat(model: str, host: str | None = None, timeout: float = 600.0) -> LLMFn:
    """Return a function that sends chat messages to a local Ollama server."""
    base = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"

    def call(messages: list[dict]) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {base}. Is it running and is the model pulled?"
            ) from exc
        return body["message"]["content"]

    return call


def ask(
    question: str,
    db_path: str | Path,
    llm: LLMFn,
    max_retries: int = 1,
    max_rows: int = 100,
) -> dict:
    """Generate SQL for a question, validate and run it, retrying once with the error."""
    messages = build_messages(question)
    sql, error = "", None
    for _ in range(max_retries + 1):
        reply = llm(messages)
        if reply.strip().upper().startswith("REFUSE"):
            return {"question": question, "sql": "", "rows": None, "error": REFUSAL}
        sql = extract_sql(reply)
        try:
            rows = run_query(db_path, validate_sql(sql), max_rows)
            return {"question": question, "sql": sql, "rows": rows, "error": None}
        except (UnsafeSQL, duckdb.Error) as exc:
            error = str(exc)
            messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That failed: {error}\nReturn one corrected SELECT statement only.",
                },
            ]
    return {"question": question, "sql": sql, "rows": None, "error": error}
