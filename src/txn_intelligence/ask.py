"""Ask a question about the transactions table in plain English."""

from __future__ import annotations

import argparse

from .nl2sql import LLMUnavailable, ask, ollama_chat


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        result = ask(args.question, args.db, ollama_chat(args.model), max_retries=args.retries)
    except LLMUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    print(f"SQL: {result['sql']}\n")
    if result["error"]:
        raise SystemExit(f"Failed: {result['error']}")
    print(result["rows"].to_string(index=False))


if __name__ == "__main__":
    main()
