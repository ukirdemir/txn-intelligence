# txn-intelligence

Fraud detection on card-style transactions, built as a small end-to-end ML product: data, models, cost-aware evaluation, a scoring API, drift monitoring and a local-LLM assistant for asking questions about the data.

> **Data:** public, synthetic transaction data only (PaySim). No real personal financial data is used or committed.

## At a glance

- **What it does:** scores each transaction before approval, picks the decision threshold that minimises an explicit expected cost on a validation period, and reports results on a later test period. It is served through FastAPI and Docker, monitored for drift, and comes with a local-LLM assistant that answers questions about the transactions through validated, read-only SQL.
- **Headline result** (last 20% of PaySim by time: 1.27M transactions, 4,254 frauds, features known before approval): gradient boosting costs 4.6 per 1,000 transactions (95% day-block interval 2.4 to 14.6) and catches 99.6% of fraud, against 12.5 for a one-line rule (amount exactly equals the sender's balance) and 334.3 for flagging nothing.
- **What to take from it:** the one-line rule gets most of the way, and the learned model adds a real but modest gain over it. That gain holds across bootstrap resamples and across retraining seeds, but its size is uncertain.
- **What not to take from it:** PaySim is synthetic and nearly separable, so the scores say more about the simulator than about real card fraud. Costs are assumed. See [Limitations](#limitations).

## Problem

Card fraud is rare but expensive: missing one fraud costs far more than reviewing a legitimate transaction, and accuracy is misleading when only a fraction of a percent of transactions are fraudulent. This project scores each transaction before approval and picks the decision threshold that minimises an explicit expected cost on a validation period, then reports results on a later test period.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

Prepare the data (PaySim CSV from Kaggle, see [data/README.md](data/README.md)), then train and evaluate:

```bash
python -m txn_intelligence.prepare --csv data/raw/paysim.csv
python -m txn_intelligence.train_baseline --db data/processed/paysim.duckdb
python -m txn_intelligence.train_compare --db data/processed/paysim.duckdb --features pre
```

Analyses (each prints a table and writes JSON to `artifacts/`; all take `--db data/processed/paysim.duckdb`):

```bash
python -m txn_intelligence.train_compare --features full   # optimistic: also uses post-transaction balances, for comparison only
python -m txn_intelligence.ablation --permutation          # which feature groups drive the score
python -m txn_intelligence.sensitivity                     # missed-fraud cost 10 to 500 instead of 100
python -m txn_intelligence.bootstrap [--method days]       # confidence intervals
python -m txn_intelligence.seeds --seeds 10                # retraining variance
python -m txn_intelligence.calibrate                       # isotonic and Platt calibration
python -m txn_intelligence.monitor [--simulate-shift amount_x3]
```

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

Cost and alert rates are much less certain than the row bootstrap suggests (gradient boosting: 4.0 to 5.2 by rows, 2.4 to 14.6 by days), because a few high-fraud-share days dominate. Recall barely changes between the two methods. The ranking is robust: models share the same resamples, and gradient boosting is cheaper than the rule, logistic regression and the MLP in 100% of replicates under both methods. The paired cost gap to the rule is 7.9 per 1k (95% interval 4.1 to 24.0 by days), so the gain over the one-line rule is real but its size is uncertain. With only about 17 test days the day-block intervals are themselves rough.

### Seed stability

Retraining with seeds 0 to 9 on the same time split (threshold re-tuned each run), mean ± standard deviation [min, max] on the test period:

| Model | Cost / 1k | Recall | PR-AUC |
|---|---|---|---|
| Gradient boosting | 4.70 ± 0.29 [4.51, 5.29] | 0.9959 ± 0.0009 | 0.9988 ± 0.0003 |
| PyTorch MLP | 24.1 ± 4.9 [17.1, 31.7] | 0.947 ± 0.017 | 0.9722 ± 0.0041 |

Gradient boosting is stable under retraining, and even its worst seed (5.29) beats the MLP's best (17.14), so the ranking does not depend on the seed. The MLP varies widely: the single-run cost of 28.3 in the results table is seed 0, near the top of its range, so read the MLP cost as roughly 17 to 32. Retraining variance for gradient boosting (cost 4.5 to 5.3) is much smaller than the uncertainty from which test days are sampled (day-block interval 2.4 to 14.6), so the choice of test period matters more than the seed. The split is fixed in time, so different train and test periods are not covered.

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

Permutation importance puts almost all of the signal in two features: `orig_would_be_emptied` (PR-AUC drop 0.994) and `amount_to_orig_balance` (0.990). The decisive pattern is a transaction whose amount exactly equals the sender's balance. Among TRANSFER and CASH_OUT transactions with a positive balance, that holds for 8,018 of 8,172 fraud cases and for none of 1,453,655 legitimate ones, while the looser "amount greater than or equal to balance" holds for 81% of legitimate cases (PaySim does not stop legitimate transactions from exceeding the balance). Importance is shared between the two correlated features, so read the individual numbers loosely. Removing the explicit features hurts only partly because the model can rebuild them from `log_amount` and `log_old_balance_orig`. Destination features, transaction type and hour add essentially nothing once balance information is present. This is a property of PaySim's simulated fraud, not evidence the approach would work on real card fraud. These are single-run numbers; see the seed and bootstrap sections for how much results move.

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

Read this with care. PaySim scores are almost binary: of 1.27M test rows, 1,268,282 sit near 0, 4,238 near 1 and only 4 in between, so the reliability curve says nothing about mid-range probabilities. Isotonic outputs exactly 0 and 1, which overstates certainty (its log loss is worse than Platt's). The calibration also held up across fraud shares (validation 0.06%, test 0.33%, thin-volume days 0.77%) because the data is nearly separable; on real data a shift in fraud share would break it. For these reasons the API keeps returning `fraud_score` and does not expose a calibrated probability.

## Serving

Train and save the serving model (gradient boosting, pre-approval features), then run the API:

```bash
python -m txn_intelligence.train_final --db data/processed/paysim.duckdb
uvicorn txn_intelligence.api:app --reload
```

The API listens on port 8000 by default; if that port is blocked on your machine, add `--port 8765`. Interactive docs are at `/docs`. Score a transaction (PowerShell, adjust the port):

```powershell
$body = '{"step": 10, "type": "TRANSFER", "amount": 181.0, "old_balance_orig": 181.0, "old_balance_dest": 0.0}'
Invoke-RestMethod -Uri http://localhost:8000/score -Method Post -ContentType "application/json" -Body $body
```

Example response for a transfer that sends the sender's whole balance:

```json
{"fraud_score": 0.9984, "flagged": true, "threshold": 0.9977, "rule_flag": true, "model_version": "hgb-pre-0.1.0"}
```

The response contains a fraud score (a ranking score, not a calibrated probability, because the model is trained with class weights), the cost-tuned decision (`flagged`), and `rule_flag`, the one-line benchmark rule, so you can see where the model and the rule disagree.

**Explanations.** Add `?explain=true` to `/score` to also get `reasons` (score versus threshold, transaction type, whether the amount equals or exceeds the sender's balance, receiver balance, and whether the one-line rule agrees) and a `note`. The reasons are built by code from the request and the decision, with no LLM. They describe signals present in the input; they are not a per-transaction attribution of what drove the model's score.

Add `&llm=true` to also get an `explanation`: a local model (Ollama; `EXPLAIN_MODEL`, `EXPLAIN_TIMEOUT` in seconds and `OLLAMA_HOST` set the model, wait time and server; the first call can be slow while the model loads) rewrites those reasons into two or three sentences. The rewrite is accepted only if it passes code checks: no number that is not in the reasons, no number words that are not in them, no causal wording ("because", "due to", "as it", "hence" and similar; a list, so some phrasings can still slip through), and wording consistent with the flagged or not-flagged decision. A rejected rewrite gets one retry with the reason fed back. If it still fails, or if Ollama is unreachable, the response falls back to the template text and says so in `explanation_source` (`llm` or `template`). The checks are deliberately strict, so a harmless rewrite may be replaced by the template. They verify numbers and decision wording only, not every possible claim, so treat the LLM text as a convenience and the `reasons` list as the source of truth.

**Docker** (the image excludes torch; train the model first so `artifacts/model.joblib` exists):

```bash
docker compose up --build
```

The API is then at `http://localhost:8765` (host port 8765 is mapped to the container's port 8000 in `docker-compose.yml`; change the left number if it clashes on your machine). A container cannot reach an Ollama server on the host unless `OLLAMA_HOST` points to it, so `llm=true` falls back to the template inside Docker by default.

The model bundle is a pickle, so scikit-learn is pinned to one exact version (`pyproject.toml`) for training, CI and the image. If you change the pin, retrain the bundle (`train_final`) before building the image, and only load bundles you created yourself.

## Drift monitoring

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

## Ask your transactions (local LLM assistant)

A local model (Ollama; small Qwen models run on CPU) turns a plain-English question into one DuckDB query, runs it safely and, when it can, writes a short answer.

```bash
ollama pull qwen2.5-coder:3b
python -m txn_intelligence.ask "How many fraud cases are there per transaction type?" [--model qwen2.5-coder:7b] [--no-answer]
```

**Safety.** Generated SQL is never trusted: it must be a single SELECT with no comments or file-access keywords, runs on a read-only connection with external access disabled, is row-capped and time-limited, and gets one retry with the error fed back. The prompt also tells the model to refuse requests that change data or are not questions about the table; that refusal is a courtesy to the user, and the actual protection is the validator and the read-only connection.

**Written answers.** For results of up to 20 rows the model writes one or two sentences, and the answer is shown only if every number in it appears in the query result or in the question. Numbers must be digits, and spelled-out numbers are rejected. If a check fails, only the SQL and the table are shown.

**Chat page.** A local page shows the answer, the SQL and the result table (127.0.0.1 only; not for Docker or public hosting, since it runs model-written SQL):

```bash
pip install -e ".[chat]"
python -m txn_intelligence.chat
```

**Evaluation.** A 30-question set with reference SQL measures execution accuracy (results are compared, not SQL text):

```bash
python -m txn_intelligence.eval_assistant --models qwen2.5-coder:3b qwen2.5-coder:7b
python -m txn_intelligence.eval_assistant --set held-out --models qwen2.5-coder:3b qwen2.5-coder:7b
```

Results on the 30-question set (temperature 0, one run, one retry allowed):

| Model | Valid SQL | Execution accuracy | Mean seconds per question |
|---|---|---|---|
| qwen2.5-coder:3b | 100.0% | 86.7% | 0.5 |
| qwen2.5-coder:7b | 100.0% | 93.3% | 3.4 |

Failures were wrong queries, not invalid SQL. Both models missed "highest number of fraud cases in any single step" (it needs a nested aggregate). The 3b model also used `/` instead of `//`, averaged instead of taking the median, and dropped the denominator in a percentage. The 7b model used the balance before the transaction where the question meant the balance after.

After adding two prompt rules (balance after/before; nested counting per step) and one example, the same 30 questions gave:

| Model | Valid SQL | Execution accuracy |
|---|---|---|
| qwen2.5-coder:3b | 100.0% | 90.0% |
| qwen2.5-coder:7b | 100.0% | 100.0% |

The rules were written after seeing the failures above, on the same questions, so this second table is optimistic: the prompt was tuned on this set. For an unbiased number, a held-out set of 12 questions was written before either model saw them and run with no prompt changes afterwards:

| Model | Valid SQL | Execution accuracy |
|---|---|---|
| qwen2.5-coder:3b | 100.0% (12/12) | 91.7% (11/12) |
| qwen2.5-coder:7b | 100.0% (12/12) | 100.0% (12/12) |

The 3b miss used `new_balance_dest` where the question said "before". Twelve questions is a small sample: 12 of 12 is consistent with a true accuracy anywhere from about 75% up, so treat this as "roughly 90% or better on simple aggregate questions", not as a precise figure. The questions are simple single-table aggregates; joins and multi-step questions were not tested. The held-out set was run a second time after the refusal rule was added, with identical results (a second look at the set, not an independent run).

## Repository layout

```
src/txn_intelligence/
  data.py, features.py, prepare.py        data layer and pre-approval features
  evaluate.py, models.py                  cost-aware metrics; PyTorch MLP and rule baselines
  train_baseline.py, train_compare.py     baseline and model comparison
  ablation.py, sensitivity.py             feature ablation; cost-ratio sensitivity
  bootstrap.py, seeds.py, calibrate.py    uncertainty, retraining variance, calibration
  train_final.py, serving.py, api.py      serving model, bundle and FastAPI service
  explain.py                              template and LLM-phrased explanations
  monitor.py                              drift monitoring
  nl2sql.py, answer.py, ask.py            SQL assistant and verified answers
  eval_assistant.py, chat.py              assistant evaluation and local chat page
tests/                  unit tests
sql/                    example analytics queries
docs/                   PRD, roadmap, model card
data/                   data instructions (data itself is gitignored)
.github/workflows/      CI (lint + tests)
Dockerfile, docker-compose.yml
```

## Product docs

- [Product requirements (PRD)](docs/PRD.md)
- [Roadmap](docs/ROADMAP.md)
- [Model card](docs/MODEL_CARD.md)

## Limitations

PaySim is synthetic and its fraud follows a simple pattern, so these results do not transfer to real fraud. Costs are assumed. Uncertainty is covered in two parts: bootstrap intervals for test-set sampling and a 10-seed study of retraining variance, both on one fixed time split, so different train and test periods are not covered. Scores are not calibrated probabilities. Drift monitoring is a batch CLI, not a scheduled service, and there is no real-time feature store. The test period is not uniform: the daily fraud share ranges from 0.05% to over 4%, so the 0.33% headline hides large day-to-day differences. The LLM assistant was evaluated on simple single-table questions with small question sets.
