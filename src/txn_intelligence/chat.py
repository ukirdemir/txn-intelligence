"""Local chat page for asking questions about the transactions table (needs the `chat` extra)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .answer import answer_question
from .nl2sql import LLMFn, LLMUnavailable, ollama_chat

EMPTY = pd.DataFrame()


def respond(question: str, db_path: str | Path, llm: LLMFn) -> tuple[str, pd.DataFrame, str]:
    """Return (sql, table, message) for one question; never raises on model or SQL errors."""
    question = (question or "").strip()
    if not question:
        return "", EMPTY, "Type a question first."
    try:
        result = answer_question(question, db_path, llm)
    except LLMUnavailable as exc:
        return "", EMPTY, str(exc)
    if result["error"]:
        return result["sql"], EMPTY, f"Could not answer: {result['error']}"
    if result["verified"]:
        return result["sql"], result["rows"], result["answer"]
    if result["unsupported"]:
        note = "Written answer withheld (it contained numbers not in the result). See the table."
    else:
        note = "No written answer for this result. See the table."
    return result["sql"], result["rows"], note


def build_app(db_path: str | Path, default_model: str, models: list[str]):
    import gradio as gr

    def run(question: str, model: str):
        return respond(question, db_path, ollama_chat(model))

    with gr.Blocks(title="Ask your transactions") as app:
        gr.Markdown(
            "# Ask your transactions\n"
            "Questions are turned into read-only SQL by a local model. "
            "Numbers in written answers are checked against the result."
        )
        model = gr.Dropdown(models, value=default_model, label="Model")
        question = gr.Textbox(label="Question", placeholder="How many fraud cases per type?")
        button = gr.Button("Ask")
        message = gr.Textbox(label="Answer", interactive=False)
        sql = gr.Code(label="SQL", language="sql", interactive=False)
        table = gr.Dataframe(label="Result", interactive=False)
        outputs = [sql, table, message]
        button.click(run, [question, model], outputs)
        question.submit(run, [question, model], outputs)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/processed/paysim.duckdb")
    parser.add_argument("--models", nargs="+", default=["qwen2.5-coder:7b", "qwen2.5-coder:3b"])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    app = build_app(args.db, args.models[0], args.models)
    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
