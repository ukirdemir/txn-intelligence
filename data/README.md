# Data

Data is never committed (see `.gitignore`). Use public or synthetic data only.

## PaySim (used by this project)

PaySim is a synthetic mobile-money transaction dataset, available on Kaggle ("Synthetic Financial Datasets For Fraud Detection"). Check the dataset's licence and terms on its Kaggle page, and record the source, version and licence in `docs/MODEL_CARD.md`.

1. Download the CSV manually from Kaggle (a free account is required).
2. Save it as `data/raw/paysim.csv`.
3. Build the validated DuckDB table:

   ```bash
   python -m txn_intelligence.prepare --csv data/raw/paysim.csv
   ```

   This writes `data/processed/paysim.duckdb` with a `transactions` table (~6.4M rows).

4. Explore with the queries in `sql/`, for example via the DuckDB CLI or `txn_intelligence.data.run_sql_file`.

## Layout

```
data/raw/        original download (gitignored)
data/processed/  DuckDB database (gitignored)
```

## Notes

- Time column `step` is an hour index (1 step = 1 hour, 743 steps in total).
- Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions. Document this in the model card, because it shapes what the model can and cannot claim.
- `is_flagged_fraud` is the output of an existing rule and is excluded from model features to avoid leakage.
