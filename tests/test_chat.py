import pytest

from txn_intelligence.chat import respond
from txn_intelligence.data import write_duckdb
from txn_intelligence.nl2sql import LLMUnavailable


@pytest.fixture
def db_path(paysim_df, tmp_path):
    return write_duckdb(paysim_df, tmp_path / "t.duckdb")


def scripted(sql: str, answer: str):
    def llm(messages):
        return sql if messages[0]["content"].startswith("You translate") else answer

    return llm


def test_empty_question(db_path):
    sql, table, msg = respond("  ", db_path, scripted("SELECT 1", "1"))
    assert sql == ""
    assert table.empty
    assert "question" in msg.lower()


def test_verified_answer(db_path, paysim_df):
    n = len(paysim_df)
    llm = scripted("SELECT COUNT(*) FROM transactions", f"There are {n} rows.")
    sql, table, msg = respond("How many?", db_path, llm)
    assert sql.startswith("SELECT")
    assert not table.empty
    assert str(n) in msg


def test_withheld_answer_still_shows_table(db_path):
    llm = scripted("SELECT COUNT(*) FROM transactions", "There are 7 rows.")
    _, table, msg = respond("How many?", db_path, llm)
    assert not table.empty
    assert "withheld" in msg


def test_unsafe_sql_reported(db_path):
    sql, table, msg = respond("x", db_path, scripted("DROP TABLE transactions", "x"))
    assert sql == "DROP TABLE transactions"
    assert table.empty
    assert msg.startswith("Could not answer")


def test_llm_unavailable_reported(db_path):
    def llm(messages):
        raise LLMUnavailable("Ollama is down")

    _, table, msg = respond("x", db_path, llm)
    assert table.empty
    assert "Ollama is down" in msg
