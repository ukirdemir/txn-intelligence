# txn-intelligence

Fraud/anomaly detection and spending insights for card transactions, built as a small end-to-end ML product: data, model, SQL analytics, cost-aware evaluation, API, and monitoring.

> **Status:** data layer, models, feature ablation, scoring API and Docker are done. The local LLM assistant and drift monitoring are done. See [docs/ROADMAP.md](docs/ROADMAP.md).
> **Data:** public or synthetic transaction data only. No real personal financial data is used or committed.

## Problem

Card fraud is rare but expensive: missing one fraud costs far more than reviewing a legitimate transaction, and accuracy is misleading when only a fraction of a percent of transactions are fraudulent. This project scores each transaction before approval and picks the decision threshold that minimises an explicit expected cost on a validation period, then reports results on a later test period.

## Results

Held-out test period (last 20% of PaySim by time: 1.27M transactions, 4,254 frauds). Features are limited to information available **before approval**. Thresholds are tuned on a separate validation period by minimising an assumed cost matrix.

| Model | PR-AUC | Precision @90% recall | Cost per 1k txns | Alerts per 1k |
|---|---|---|---|---|
| Flag nothing | n/a | n/a | 334.3 | 0 |
| One-line rule: TRANSFER/CASH_OUT with amount exactly equal to sender's balance | 0.972 | 1.000 | 12.5 | 3.25 |
| Logistic regression | 0.817 | 0.422 | 77.4 | 4.89 |
| PyTorch MLP | 0.970 | 0.908 | 28.3 | 3.67 |
| Gradient boosting | 0.999 | 1.000 | 4.6 | 3.33 |

**Read these with care.** PaySim is synthetic and its fraud follows a simple script (full-balance transfers and cash-outs), so near-perfect scores say more about the simulator than about real-world fraud. A first run that also used post-transaction balances scored even higher (PR-AUC 1.000 for both the MLP and gradient boosting) and was discarded as unrealistic, because those balances are not known before approval. Gradient boosting beats the MLP here, which is common on tabular data. The one-line rule in the second row beats both the logistic regression and the MLP on cost. Gradient boosting is better still (cost 4.6 against 12.5, 16 missed frauds against 118), so the learned model adds real but modest value over the rule. A looser rule (amount greater than or equal to the balance) fails completely, because it matches 81% of legitimate transactions. Costs use placeholder values (see `docs/PRD.md`).

### Uncertainty (bootstrap)

95% percentile intervals from 1,000 bootstrap replicates of the test period (thresholds fixed from validation; models not retrained, so only test-set sampling is covered). Rows are resampled in two ways: individually, and in whole-day blocks. Because PaySim days differ sharply in volume and fraud share, the day-block intervals are the ones to trust.

| Model | Cost / 1k, rows | Cost / 1k, day blocks | Recall, day blocks | PR-AUC, day blocks |
|---|---|---|---|---|
| Rule: amount equals balance | 12.5 [11.0, 14.2] | 12.5 [6.6, 38.6] | 0.972 [0.965, 0.980] | 0.972 [0.965, 0.980] |
| Logistic regression | 77.4 [73.2, 81.8] | 77.4 [46.7, 213.8] | 0.809 [0.798, 0.820] | 0.817 [0.762, 0.898] |
| PyTorch MLP | 28.3 [25.7, 30.9] | 28.3 [16.7, 78.7] | 0.933 [0.926, 0.939] | 0.970 [0.954, 0.986] |
| Gradient boosting | 4.6 [4.0, 5.2] | 4.6 [2.4, 14.6] | 0.996 [0.994, 0.998] | 0.999 [0.998, 1.000] |

Cost and alert rates are much less certain than the row bootstrap suggests (gradient boosting: 4.0 to 5.2 by rows, 2.4 to 14.6 by days), because a few high-fraud-share days dominate. Recall barely changes between the two methods. The ranking is robust: models share the same resamples, and gradient boosting is cheaper than the rule, logistic regression and the MLP in 100% of replicates under both methods. The paired cost gap to the rule is 7.9 per 1k (95% interval 4.1 to 24.0 by days), so the gain over the one-line rule is real but its size is uncertain. With only about 17 test days the day-block intervals are themselves rough. Run with `python -m txn_intelligence.bootstrap [--method days]`.

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

Permutation importance puts almost all of the signal in two features: `orig_would_be_emptied` (PR-AUC drop 0.994) and `amount_to_orig_balance` (0.990). The decisive pattern is a transaction whose amount exactly equals the sender's balance. Among TRANSFER and CASH_OUT transactions with a positive balance, that holds for 8,018 of 8,172 fraud cases and for none of 1,453,655 legitimate ones, while the looser "amount greater than or equal to balance" holds for 81% of legitimate cases (PaySim does not stop legitimate transactions from exceeding the balance). Importance is shared between the two correlated features, so read the individual numbers loosely. Removing the explicit features hurts only partly because the model can rebuild them from `log_amount` and `log_old_balance_orig`. Destination features, transaction type and hour add essentially nothing once balance information is present. This is a property of PaySim's simulated fraud, not evidence the approach would work on real card fraud. Single training run (the bootstrap above covers test sampling only).

### Cost sensitivity

Cost per 1,000 transactions when a missed fraud costs 10, 100 or 500 (false alert fixed at 5; thresholds re-tuned on validation each time):

| Missed-fraud cost (ratio) | One-line rule | Logistic regression | Gradient boosting |
|---|---|---|---|
| 10 (2:1) | 4.2 | 17.7 | 3.5 |
| 100 (20:1) | 12.5 | 77.4 | 4.6 |
| 500 (100:1) | 49.6 | 174.4 | 9.6 |

Gradient boosting has the lowest cost at every ratio from 2:1 to 100:1, but its advantage over the rule shrinks as misses get cheaper (17% lower at 2:1, 81% lower at 100:1). It flags the same transactions at every ratio because PaySim fraud is almost perfectly separable, so on real data the threshold would matter far more.

### Calibration

The served score is a ranking score: the model is trained with class weights, so it overstates the fraud rate (mean score 0.0052 on the test period against an observed 0.0033). `calibrate.py` fits isotonic and Platt maps on the validation period and checks them on test:

| Method | Brier | ECE | Mean predicted (observed 0.00334) |
|---|---|---|---|
| Raw score | 0.000056 | 0.00189 | 0.00522 |
| Isotonic | 0.000011 | 0.00001 | 0.00333 |
| Platt | 0.000034 | 0.00005 | 0.00339 |

Calibration fixes the average level (and by period: raw scores overstate the fraud rate 3.7 times on high-volume days, isotonic is within 2%). It does not change ranking, recall or PR-AUC, and the validation-tuned threshold gives the same cost (4.6 per 1k, 16 missed). A threshold taken from the cost matrix alone (flag above 5 / (5 + 100 − 1) = 0.048) gave 4.3 per 1k and 12 missed on isotonic probabilities, but that difference is 4 transactions and well inside the bootstrap intervals, so it is not evidence of an improvement.

Read this with care. PaySim scores are almost binary: of 1.27M test rows, 1,268,282 sit near 0, 4,238 near 1 and only 4 in between, so the reliability curve says nothing about mid-range probabilities. Isotonic outputs exactly 0 and 1, which overstates certainty (its log loss is worse than Platt's). The calibration also held up across fraud shares (validation 0.06%, test 0.33%, thin-volume days 0.77%) because the data is nearly separable; on real data a shift in fraud share would break it. For these reasons the API keeps returning `fraud_score` and does not expose a calibrated probability. Run with `python -m txn_intelligence.calibrate`.

### Drift monitoring

`monitor.py` compares each day of the test period with the validation period (the reference) using PSI and the KS statistic on every model input and on the score, plus alerts per 1,000. It needs no labels. PSI bands (under 0.1 stable, 0.1 to 0.25 watch, over 0.25 investigate) are conventions, and KS is reported as a statistic, not a p-value.

```bash
python -m txn_intelligence.monitor
python -m txn_intelligence.monitor --simulate-shift amount_x3
python -m txn_intelligence.monitor --simulate-shift type_mix
```

**PaySim is not stationary.** Fraud arrives at a steady 220 to 316 cases a day, but legitimate volume collapses on some days (days 2 to 4 and 17 to 30), so the daily fraud share swings from 0.05% to over 4% and the last day is all fraud (`sql/02_fraud_rate_by_day.sql`). In the test period, days 15 and 16 (375k and 421k rows) look like the reference (score PSI at most 0.001), while days 17 to 29 (8k to 59k rows) flag 4 to 32 alerts per 1,000 against 0.61 in the reference. The monitor reports this correctly. Almost every day also shows `investigate` because the hour-of-day mix changes, which makes the overall status a blunt signal on this dataset: read the score PSI and the alert rate first. I expected little drift before running this and was wrong.

To check that the monitor detects problems at all, one day (day 29) is artificially shifted and compared with the same day unshifted:

| Injected shift | Score PSI | Alerts per 1k | Largest feature PSI increases |
|---|---|---|---|
| Amounts multiplied by 3 | 0.150 to 0.340 | 23.31 to 0.18 | `log_amount` 0.013 to 0.577 |
| Half the rows set to TRANSFER | 0.150 to 0.161 | 23.31 to 23.31 | `type_TRANSFER` 0.000 to 1.176 |

Tripling amounts breaks the "amount equals balance" pattern, so the model silently stops flagging; both the score drift and the input drift show it. Changing the type mix does not move the score or alerts, because the model barely uses type, yet the input PSI catches it. Monitoring inputs and scores together therefore matters. These shifts are artificial, and PSI and KS detect change in inputs and scores, not loss of accuracy; real monitoring also needs delayed labels. The last day has too few rows (282) and is skipped by the 1,000-row minimum.

Example `/score` response for a transfer that sends the sender's whole balance:

```json
{"fraud_score": 0.9984, "flagged": true, "threshold": 0.9977, "rule_flag": true, "model_version": "hgb-pre-0.1.0"}
```

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

Cost sensitivity (how the threshold and model ranking change if a missed fraud costs 10 to 500 instead of 100):

```bash
python -m txn_intelligence.sensitivity --db data/processed/paysim.duckdb
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

Add `?explain=true` to `/score` to also get `reasons` (score versus threshold, transaction type, whether the amount equals or exceeds the sender's balance, receiver balance, and whether the one-line rule agrees) and a `note`. The reasons are built by code from the request and the decision, with no LLM. They describe signals present in the input; they are not a per-transaction attribution of what drove the model's score.

The response contains a fraud score (a ranking score, not a calibrated probability, because the model is trained with class weights), the cost-tuned decision (`flagged`), and `rule_flag`, the one-line benchmark rule, so you can see where the model and the rule disagree. Interactive docs are at `http://localhost:8000/docs`.

With Docker (the image excludes torch; train the model first so `artifacts/model.joblib` exists):

```bash
docker compose up --build
```

The API is then at `http://localhost:8765` (host port 8765 is mapped to the container's port 8000 in `docker-compose.yml`; change the left number if it clashes on your machine).

The model bundle is a pickle, so scikit-learn is pinned to one exact version (`pyproject.toml`) for training, CI and the image. If you change the pin, retrain the bundle (`train_final`) before building the image, and only load bundles you created yourself.

### Ask your transactions (stage 1: question to SQL)

A local LLM (Ollama, CPU is enough for small Qwen models) turns a plain-English question into one DuckDB query. Generated SQL is never trusted: it must be a single SELECT with no comments or file-access keywords, runs on a read-only connection with external access disabled, is row-capped and time-limited, and gets one retry with the error fed back.

```bash
ollama pull qwen2.5-coder:3b
python -m txn_intelligence.ask "How many fraud cases are there per transaction type?"
```

A 30-question evaluation set with reference SQL measures execution accuracy (results compared, not SQL text):

```bash
python -m txn_intelligence.eval_assistant --models qwen2.5-coder:3b qwen2.5-coder:7b
```

To use the local chat page (127.0.0.1 only; not for Docker or public hosting, since it runs model-written SQL):

```bash
pip install -e ".[chat]"
python -m txn_intelligence.chat
```

Results (30 questions, temperature 0, one run, one retry allowed):

| Model | Valid SQL | Execution accuracy | Mean seconds per question |
|---|---|---|---|
| qwen2.5-coder:3b | 100.0% | 86.7% | 0.5 |
| qwen2.5-coder:7b | 100.0% | 93.3% | 3.4 |

After adding two prompt rules (balance after/before; nested counting per step) and one example, the same 30 questions gave:

| Model | Valid SQL | Execution accuracy |
|---|---|---|
| qwen2.5-coder:3b | 100.0% | 90.0% |
| qwen2.5-coder:7b | 100.0% | 100.0% |

These rules were written after seeing the failures above, on the same questions, so the second table is optimistic: it measures the prompt on the set it was tuned against. A fresh held-out set is needed for an unbiased number.

On a held-out set of 12 questions written before either model saw them, with no prompt changes afterwards (`--set held-out`, run once):

| Model | Valid SQL | Execution accuracy |
|---|---|---|
| qwen2.5-coder:3b | 100.0% (12/12) | 91.7% (11/12) |
| qwen2.5-coder:7b | 100.0% (12/12) | 100.0% (12/12) |

The 3b miss used `new_balance_dest` where the question said "before". Twelve questions is a small sample: 12 of 12 is consistent with a true accuracy anywhere from about 75% up, so treat this as "roughly 90% or better on simple aggregate questions", not as a precise figure. The questions are simple single-table aggregates; joins and multi-step questions were not tested.

A later prompt change added a rule to refuse requests that modify data or are not questions about the table. The held-out set was run a second time after that change with identical results (this is a second look at the set, not an independent run). Both models refuse "Delete all fraud transactions from the table." Refusal is a courtesy to the user; the actual protection is the validator and the read-only connection.

Failures in the first run were wrong queries, not invalid SQL. Both models missed "highest number of fraud cases in any single step" (it needs a nested aggregate). The 3b model also used `/` instead of `//`, averaged instead of taking the median, and dropped the denominator in a percentage. The 7b model used the balance before the transaction where the question meant the balance after. The eval set is small and the prompt was not tuned against it, so differences of one or two questions are noise.

## What's inside

- **Model:** baseline plus a PyTorch classifier, evaluated with precision-recall metrics and a cost-based decision threshold.
- **LLM layer (in progress):** a local-model assistant that answers questions about the transactions through validated, read-only SQL (stage 1 written; see above).
- **SQL analytics:** DuckDB queries over the transaction table (`sql/`).
- **Serving:** FastAPI service in Docker with a `/score` endpoint.
- **Monitoring:** batch drift checks on score, feature distributions and alert rate (`monitor.py`).

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

PaySim is synthetic and its fraud follows a simple pattern, so these results do not transfer to real fraud. Single training run; confidence intervals cover test-set sampling only, not training or threshold variance. Costs are assumed. Scores are not calibrated probabilities. Drift monitoring is a batch CLI, not a scheduled service, and there is no real-time feature store. The test period is not uniform: the daily fraud share ranges from 0.05% to over 4%, so the 0.33% headline hides large day-to-day differences.
