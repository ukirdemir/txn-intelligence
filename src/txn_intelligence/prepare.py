"""Convert the raw PaySim CSV into a validated DuckDB table."""

from __future__ import annotations

import argparse

from .data import load_paysim, write_duckdb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the raw PaySim CSV file")
    parser.add_argument(
        "--db", default="data/processed/paysim.duckdb", help="Output DuckDB file path"
    )
    args = parser.parse_args(argv)

    df = load_paysim(args.csv)
    path = write_duckdb(df, args.db)
    fraud_rate = 100 * df["is_fraud"].mean()
    print(f"Wrote {len(df):,} rows to {path} (fraud rate {fraud_rate:.3f}%)")


if __name__ == "__main__":
    main()
