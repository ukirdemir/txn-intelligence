import pandas as pd
import pytest

from txn_intelligence.answer import (
    answer_question,
    build_answer_messages,
    numbers_in,
    unsupported_numbers,
)
from txn_intelligence.data import write_duckdb


@pytest.fixture
def db_path(paysim_df, tmp_path):
    return write_duckdb(paysim_df, tmp_path / "t.duckdb")


def scripted(sql: str, answer: str):
    def llm(messages):
        return sql if messages[0]["content"].startswith("You translate") else answer

    return llm


def test_numbers_in():
    assert numbers_in("There are 1,234 rows and 3.50 percent") == [(1234.0, 0), (3.5, 2)]


def test_supported_numbers_pass():
    rows = pd.DataFrame({"n": [1234], "r": [3.4987]})
    assert unsupported_numbers("1,234 rows, rate 3.50", rows) == []


def test_unsupported_number_found():
    rows = pd.DataFrame({"n": [1234]})
    assert unsupported_numbers("There are 1,235 rows", rows) == [1235.0]


def test_number_words_rejected():
    rows = pd.DataFrame({"n": [40]})
    assert unsupported_numbers("The highest is forty", rows) == ["forty"]


def test_numbers_from_question_allowed():
    rows = pd.DataFrame({"n": [5]})
    assert unsupported_numbers("5 rows above 100000", rows, "amount above 100000?") == []


def test_build_answer_messages_contains_table():
    msgs = build_answer_messages("q?", pd.DataFrame({"n": [42]}))
    assert "42" in msgs[1]["content"]


def test_answer_verified(db_path, paysim_df):
    n = len(paysim_df)
    llm = scripted("SELECT COUNT(*) FROM transactions", f"There are {n} transactions.")
    out = answer_question("How many?", db_path, llm)
    assert out["verified"]
    assert str(n) in out["answer"]


def test_answer_withheld_on_wrong_number(db_path):
    llm = scripted("SELECT COUNT(*) FROM transactions", "There are 7 transactions.")
    out = answer_question("How many?", db_path, llm)
    assert not out["verified"]
    assert out["answer"] is None
    assert out["unsupported"] == [7.0]
    assert out["rows"] is not None


def test_empty_answer_not_verified(db_path):
    llm = scripted("SELECT COUNT(*) FROM transactions", "   ")
    out = answer_question("How many?", db_path, llm)
    assert not out["verified"]
    assert out["answer"] is None


def test_no_answer_on_sql_error(db_path):
    llm = scripted("SELECT nope FROM transactions", "irrelevant")
    out = answer_question("x", db_path, llm, max_retries=0)
    assert out["error"] is not None
    assert out["answer"] is None


def test_no_answer_for_large_results(db_path):
    llm = scripted("SELECT * FROM transactions", "irrelevant")
    out = answer_question("x", db_path, llm)
    assert out["answer"] is None
    assert out["error"] is None
