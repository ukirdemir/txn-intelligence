"""Turn a query result into a short written answer, checking every number against the data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .nl2sql import LLMFn, ask

MAX_ANSWER_ROWS = 20
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty"
    r"|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen)\b",
    re.IGNORECASE,
)

ANSWER_PROMPT = """You answer a question using only the query result below.

Rules:
- Write one or two plain sentences.
- Use only numbers that appear in the result. Do not compute new numbers.
- Write numbers as digits, never as words, and with no abbreviations such as K or M.
- Do not mention SQL.
"""


def _values(rows: pd.DataFrame) -> list[float]:
    out = []
    for value in rows.to_numpy().ravel():
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def numbers_in(text: str) -> list[tuple[float, int]]:
    """Return (value, decimals) for every number written in the text."""
    found = []
    for token in NUMBER.findall(text):
        clean = token.replace(",", "").rstrip(".")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        found.append((float(clean), decimals))
    return found


def unsupported_numbers(answer: str, rows: pd.DataFrame, question: str = "") -> list:
    """Digits matching nothing in the result or question, plus any number words."""
    allowed = _values(rows)
    allowed += [value for value, _ in numbers_in(question)]
    bad: list = []
    for value, decimals in numbers_in(answer):
        if not any(round(v, decimals) == value for v in allowed):
            bad.append(value)
    bad += [word.lower() for word in NUMBER_WORDS.findall(answer)]
    return bad


def build_answer_messages(question: str, rows: pd.DataFrame) -> list[dict]:
    table = rows.to_string(index=False)
    return [
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nQuery result:\n{table}"},
    ]


def answer_question(
    question: str, db_path: str | Path, llm: LLMFn, max_retries: int = 1
) -> dict:
    """Run the question, then write an answer only if every number in it is supported."""
    result = ask(question, db_path, llm, max_retries=max_retries)
    result = {**result, "answer": None, "verified": False, "unsupported": []}
    rows = result["rows"]
    if result["error"] or rows is None or rows.empty or len(rows) > MAX_ANSWER_ROWS:
        return result
    answer = llm(build_answer_messages(question, rows)).strip()
    bad = unsupported_numbers(answer, rows, question)
    result["unsupported"] = bad
    result["raw_answer"] = answer
    if answer and not bad:
        result["answer"] = answer
        result["verified"] = True
    return result
