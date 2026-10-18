import duckdb
import pandas as pd
import pytest

from txn_intelligence.data import write_duckdb
from txn_intelligence.eval_assistant import (
    EVAL_SET,
    evaluate_models,
    format_table,
    results_match,
)
from txn_intelligence.nl2sql import (
    LLMUnavailable,
    UnsafeSQL,
    ask,
    build_messages,
    extract_sql,
    ollama_chat,
    run_query,
    validate_sql,
)


@pytest.fixture
def db_path(paysim_df, tmp_path):
    return write_duckdb(paysim_df, tmp_path / "t.duckdb")


def fixed(reply: str):
    return lambda messages: reply


def test_extract_sql_from_fence():
    assert extract_sql("Sure:\n```sql\nSELECT 1\n```\nDone") == "SELECT 1"


def test_extract_sql_drops_leading_prose():
    assert extract_sql("Here you go: SELECT COUNT(*) FROM transactions") == (
        "SELECT COUNT(*) FROM transactions"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM transactions",
        "SELECT COUNT(*) FROM transactions;",
        "WITH t AS (SELECT * FROM transactions) SELECT COUNT(*) FROM t",
    ],
)
def test_validate_accepts(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DROP TABLE transactions",
        "SELECT 1; SELECT 2",
        "SELECT 1 -- hi",
        "SELECT 1 /* hi */",
        "INSERT INTO transactions VALUES (1)",
        "SELECT * FROM read_csv('x.csv')",
        "SELECT * FROM read_parquet('x.parquet')",
        "ATTACH 'x.db'",
        "COPY transactions TO 'x.csv'",
        "SELECT 1 WHERE 1 = 1 OR (SELECT COUNT(*) FROM glob('*')) > 0",
    ],
)
def test_validate_rejects(sql):
    with pytest.raises(UnsafeSQL):
        validate_sql(sql)


def test_run_query_row_limit(db_path):
    out = run_query(db_path, "SELECT * FROM transactions", max_rows=7)
    assert len(out) == 7


def test_run_query_blocks_external_files(db_path, tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a\n1\n")
    with pytest.raises(duckdb.Error):
        run_query(db_path, f"SELECT * FROM read_csv_auto('{csv.as_posix()}')")


def test_ask_success(db_path, paysim_df):
    out = ask("count", db_path, fixed("SELECT COUNT(*) FROM transactions"))
    assert out["error"] is None
    assert int(out["rows"].iloc[0, 0]) == len(paysim_df)


def test_ask_unsafe_not_executed(db_path):
    out = ask("x", db_path, fixed("DROP TABLE transactions"), max_retries=0)
    assert out["error"] is not None
    assert out["rows"] is None
    assert run_query(db_path, "SELECT COUNT(*) FROM transactions").iloc[0, 0] > 0


def test_ask_retries_after_failure(db_path):
    replies = iter(["SELECT nope FROM transactions", "SELECT COUNT(*) FROM transactions"])
    seen = []

    def llm(messages):
        seen.append(messages)
        return next(replies)

    out = ask("count", db_path, llm, max_retries=1)
    assert out["error"] is None
    assert len(seen) == 2
    assert "That failed" in seen[1][-1]["content"]


def test_ask_gives_up(db_path):
    out = ask("x", db_path, fixed("SELECT nope FROM transactions"), max_retries=1)
    assert out["error"] is not None
    assert out["rows"] is None


def test_eval_questions_unique():
    questions = [i["question"] for i in EVAL_SET]
    assert len(EVAL_SET) == 30
    assert len(set(questions)) == len(questions)


def test_reference_sql_valid_and_runs(db_path):
    for item in EVAL_SET:
        run_query(db_path, validate_sql(item["sql"]), max_rows=1000)


def test_results_match():
    a = pd.DataFrame({"x": ["a", "b"], "y": [1.001, 2.0]})
    b = pd.DataFrame({"p": ["b", "a"], "q": [2.0, 1.0]})
    c = pd.DataFrame({"p": ["b", "a"], "q": [2.0, 9.0]})
    assert results_match(a, b)
    assert not results_match(a, c)


def test_evaluate_models(db_path):
    reference = {i["question"]: i["sql"] for i in EVAL_SET}

    def perfect(messages):
        return reference[messages[1]["content"].removeprefix("Question: ")]

    def wrong(messages):
        return "SELECT -1"

    out = evaluate_models(db_path, {"perfect": perfect, "wrong": wrong}, limit=5)
    assert out["perfect"]["execution_accuracy"] == 1.0
    assert out["wrong"]["valid_sql_rate"] == 1.0
    assert out["wrong"]["execution_accuracy"] == 0.0
    assert out["perfect"]["n_questions"] == 5


def test_format_table(db_path):
    out = evaluate_models(db_path, {"m": fixed("SELECT -1")}, limit=2)
    table = format_table(out)
    assert "| m | 2 |" in table
    assert table.startswith("| Model")


def test_build_messages():
    msgs = build_messages("hi")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "Question: hi"


def test_ollama_unreachable():
    llm = ollama_chat("x", host="http://127.0.0.1:9", timeout=2.0)
    with pytest.raises(LLMUnavailable):
        llm(build_messages("hi"))
