# txn-intelligence

Fraud/anomaly detection and spending insights for card transactions, built as a small end-to-end ML product: data, model, SQL analytics, cost-aware evaluation, API, and monitoring.

> **Status:** early scaffold. Sections marked `TODO` are filled in as milestones in [docs/ROADMAP.md](docs/ROADMAP.md) are completed.
> **Data:** public or synthetic transaction data only. No real personal financial data is used or committed.

## Problem

TODO (2-3 sentences): who loses money to fraud, why accuracy is the wrong metric on imbalanced data, and what decision the model supports (block / review / allow).

## Results

TODO: replace with real numbers once the baseline and model are trained.

Held-out test period (last 20% of PaySim by time: 1.27M transactions, 4,254 frauds). Features are limited to information available **before approval**. Thresholds are tuned on a separate validation period by minimising an assumed cost matrix.

| Model | PR-AUC | Precision @90% recall | Cost per 1k txns | Alerts per 1k |
|---|---|---|---|---|
| Flag nothing | n/a | n/a | 334.3 | 0 |
| One-line rule: TRANSFER/CASH_OUT with amount exactly equal to sender's balance | 0.972 | 1.000 | 12.5 | 3.25 |
| Logistic regression | 0.817 | 0.422 | 77.4 | 4.89 |
| PyTorch MLP | 0.970 | 0.908 | 28.3 | 3.67 |
| Gradient boosting | 0.999 | 1.000 | 4.6 | 3.33 |

**Read these with care.** PaySim is synthetic and its fraud follows a simple script (full-balance transfers and cash-outs), so near-perfect scores say more about the simulator than about real-world fraud. A first run that also used post-transaction balances scored even higher (PR-AUC 1.000 for both the MLP and gradient boosting) and was discarded as unrealistic, because those balances are not known before approval. Gradient boosting beats the MLP here, which is common on tabular data. The one-line rule in the second row beats both the logistic regression and the MLP on cost. Gradient boosting is better still (cost 4.6 against 12.5, 16 missed frauds against 118), so the learned model adds real but modest value over the rule. A looser rule (amount greater than or equal to the balance) fails completely, because it matches 81% of legitimate transactions. Costs use placeholder values (see `docs/PRD.md`).

### What drives the score (feature ablation)

Gradient boosting, test period, pre-approval features, with feature groups removed:

| Variant | PR-AUC | Cost per 1k |
|---|---|---|
| All 12 features | 0.999 | 4.6 |
| Without destination-account features | 1.000 | 5.1 |
| Without transaction type | 0.999 | 4.6 |
| Without the two explicit account-emptying features | 0.922 | 55.9 |
| Without any origin-balance information | 0.526 | 164.9 |
| Only amount, hour and type | 0.263 | 300.8 |

Permutation importance puts almost all of the signal in two features: `orig_would_be_emptied` (PR-AUC drop 0.994) and `amount_to_orig_balance` (0.990). The decisive pattern is a transaction whose amount exactly equals the sender's balance. Among TRANSFER and CASH_OUT transactions with a positive balance, that holds for 8,018 of 8,172 fraud cases and for none of 1,453,655 legitimate ones, while the looser "amount greater than or equal to balance" holds for 81% of legitimate cases (PaySim does not stop legitimate transactions from exceeding the balance). Importance is shared between the two correlated features, so read the individual numbers loosely. Removing the explicit features hurts only partly because the model can rebuild them from `log_amount` and `log_old_balance_orig`. Destination features, transaction type and hour add essentially nothing once balance information is present. This is a property of PaySim's simulated fraud, not evidence the approach would work on real card fraud. Single run, no confidence intervals yet.

Include a demo GIF or screenshot of the API / dashboard here.

Compare logistic regression, gradient boosting and the PyTorch MLP on identical splits:

```bash
# realistic: only information available before approval (default, headline result)
python -m txn_intelligence.train_compare --db data/processed/paysim.duckdb --features pre

# optimistic: also uses post-transaction balances (shown for comparison only)
python -m txn_intelligence.train_compare --db data/processed/paysim.duckdb --features full
```

Feature ablation (which feature groups drive the scores):

```bash
python -m txn_intelligence.ablation --db data/processed/paysim.duckdb --permutation
```

Train and save the serving model (gradient boosting, pre-approval features), then run the API:

```bash
python -m txn_intelligence.train_final --db data/processed/paysim.duckdb
uvicorn txn_intelligence.api:app --reload
```

Score a transaction (PowerShell):

```powershell
$body = '{"step": 10, "type": "TRANSFER", "amount": 181.0, "old_balance_orig": 181.0, "old_balance_dest": 0.0}'
Invoke-RestMethod -Uri http://localhost:8000/score -Method Post -ContentType "application/json" -Body $body
```

The response contains a fraud score (a ranking score, not a calibrated probability, because the model is trained with class weights), the cost-tuned decision (`flagged`), and `rule_flag`, the one-line benchmark rule, so you can see where the model and the rule disagree. Interactive docs are at `http://localhost:8000/docs`.

With Docker (the image excludes torch; train the model first so `artifacts/model.joblib` exists):

```bash
docker compose up --build
```

The API is then at `http://localhost:8765` (host port 8765 is mapped to the container's port 8000 in `docker-compose.yml`; change the left number if it clashes on your machine).

The model bundle is a pickle: train it with the same scikit-learn version the image installs, and only load bundles you created yourself.

## What's inside

- **Model:** baseline plus a PyTorch classifier, evaluated with precision-recall metrics and a cost-based decision threshold.
- **LLM layer:** transaction categorization and a natural-language "ask your spending" assistant (local model, optional).
- **SQL analytics:** DuckDB queries over the transaction table (`sql/`).
- **Serving:** FastAPI service in Docker with a `/score` endpoint.
- **Monitoring:** drift checks on score and feature distributions (TODO).

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
uvicorn txn_intelligence.api:app --reload
# or
docker compose up --build
```

Prepare the data (PaySim CSV from Kaggle, see [data/README.md](data/README.md)):

```bash
python -m txn_intelligence.prepare --csv data/raw/paysim.csv
```

Train and evaluate the baseline (threshold tuned on validation, results on a held-out test period):

```bash
python -m txn_intelligence.train_baseline --db data/processed/paysim.duckdb
```

## Repository layout

```
src/txn_intelligence/   package code (config, data, features, prepare, evaluate, api)
tests/                  unit tests
sql/                    example analytics queries
docs/                   PRD, roadmap, model card
data/                   data instructions (data itself is gitignored)
.github/workflows/      CI (lint + tests)
```

## Product docs

- [Product requirements (PRD)](docs/PRD.md)
- [Roadmap](docs/ROADMAP.md)
- [Model card](docs/MODEL_CARD.md)

## Limitations

TODO: be explicit. Synthetic data, class imbalance assumptions, no real-time feature store, etc.
