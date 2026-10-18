# Roadmap

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Repo scaffold, CI, docs templates | Done |
| M1 | PaySim loader and validation, feature engineering, DuckDB prepare command, SQL analytics queries, tests | Done |
| M2 | Baseline model (logistic regression) with PR-AUC and cost-based threshold (tuned on validation, reported on test) | Done |
| M3 | PyTorch MLP vs logistic baseline vs gradient boosting, precision at fixed recall, error analysis | Done (results and feature ablation in README) |
| M4 | Train and save the model, FastAPI `/score` endpoint (model and rule outputs), Docker image | Done and verified locally (API scores a draining transfer 0.998 and flags it, a small payment 0.002). Docker image built and served a `/score` request (200 OK). Load test still to do |
| M5 | Drift monitoring (PSI or KS on features and scores) | TODO |
| M6 | LLM categorization and "ask your spending" assistant with eval set | TODO |
| M7 | Final write-up: results table, limitations, demo GIF | TODO |

## What I would do next and why

TODO: fill in after M3. Prioritise by expected impact on the cost metric.
