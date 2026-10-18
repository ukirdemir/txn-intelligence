"""Ask a question about the transactions table in plain English."""

from __future__ import annotations

import argparse

from .answer import answer_question
from .nl2sql import LLMUnavailable, ollama_chat
from .nl2sql import ask as ask_sql


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-answer", action="store_true", help="show only SQL and table")
    args = parser.parse_args(argv)

    llm = ollama_chat(args.model)
    try:
        if args.no_answer:
            result = {**ask_sql(args.question, args.db, llm, max_retries=args.retries)}
            result["answer"], result["unsupported"] = None, []
        else:
            result = answer_question(args.question, args.db, llm, max_retries=args.retries)
    except LLMUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    print(f"SQL: {result['sql']}\n")
    if result["error"]:
        raise SystemExit(f"Failed: {result['error']}")
    print(result["rows"].to_string(index=False))
    if result["answer"]:
        print(f"\nAnswer (numbers verified): {result['answer']}")
    elif result["unsupported"]:
        print(f"\nAnswer withheld: it contained numbers not in the result: {result['unsupported']}")
    elif not args.no_answer:
        raw = result.get("raw_answer", "")
        print(f"\nNo verified answer. Model reply: {raw!r}")


if __name__ == "__main__":
    main()
