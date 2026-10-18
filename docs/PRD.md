# Product Requirements Document: Transaction Intelligence

| | |
|---|---|
| Owner | Umut Kirdemir |
| Status | Draft, updated with results from milestones M0-M4 |
| Last updated | 2026-10-18 |

This is a portfolio project on public, synthetic data (PaySim). Cost figures are assumptions, and nothing here describes any real company's systems or losses.

## 1. Problem

Card fraud is rare but expensive. A missed fraud loses the transaction amount and customer trust, while a false alert blocks or delays a legitimate payment and adds review work. Because fraud is a tiny share of traffic (0.33% in the test period), accuracy is meaningless: a model that flags nothing is 99.7% accurate. The product question is: which transactions should we flag before approval so that the total cost of missed fraud, false alerts and review effort is lowest?

## 2. Users and jobs to be done

| User | Job to be done | Pain today |
|---|---|---|
| Fraud analyst | Work a short, well-ranked review queue | Too many false alerts, little ranking signal |
| Product owner for risk | Tune the trade-off between fraud loss and customer friction | No explicit cost view of the threshold |
| Customer | Have legitimate payments approved without delay | Blocked or delayed payments |
| Engineer integrating the service | Call one endpoint at approval time and get a decision plus a reason to trust it | Black-box scores with no benchmark |

## 3. Goals and non-goals

**Goals**
- Reduce expected cost per 1,000 transactions well below the "flag nothing" policy and below a simple rule benchmark.
- Use only information available before approval.
- Serve decisions through an API with a documented contract.
- Show where the model beats a transparent rule and where it does not.

**Non-goals**
- Real-time feature store, streaming ingestion or production-grade infrastructure.
- Use of any real customer data.
- Claims about real-world fraud performance.

## 4. Success metrics and current status

All numbers are on the held-out test period (last 20% of PaySim by time: 1,272,524 transactions, 4,254 frauds), with the threshold tuned on a separate validation period.

| Metric | Target | Current | How measured |
|---|---|---|---|
| Expected cost per 1k transactions | Below the rule benchmark (12.5) | 4.6 (gradient boosting) | `evaluate.py` cost matrix |
| Reduction vs "flag nothing" (334.3) | Above 90% | 98.6% | Same |
| PR-AUC | Above the rule benchmark (0.972) | 0.999 | Average precision on test |
| Missed frauds | Fewer than the rule benchmark (118) | 16 of 4,254 | Count at tuned threshold |
| Alerts per 1k transactions | Close to the true fraud rate (3.34) | 3.33 | Flag rate at tuned threshold |
| API p95 latency | Under 100 ms (to be confirmed) | Not measured | Load test (planned) |
| Assistant answer accuracy | Above 80% on an SQL-verified question set | Not built | Evaluation set (planned, M6) |

## 5. Decision framework

The model outputs a score. A transaction is flagged when the score reaches a threshold, and the threshold is chosen by minimising expected cost on validation data.

Assumed cost matrix (`src/txn_intelligence/config.py`):

| Outcome | Cost | Reasoning |
|---|---|---|
| Missed fraud | 100 | Average loss plus downstream handling (assumed) |
| False alert | 5 | Review time and customer friction (assumed) |
| Flagged real fraud | 1 | Review effort that is still worth spending |

The 20:1 ratio between a missed fraud and a false alert drives the threshold. It is an assumption, not a measurement. A sweep from 2:1 to 100:1 (section 6) shows how much the choice matters.

## 6. What the results say

- A one-line rule (TRANSFER or CASH_OUT where the amount exactly equals the sender's balance) already reaches PR-AUC 0.972 and cost 12.5, and it beats logistic regression (77.4) and the neural network (28.3).
- Gradient boosting improves on the rule (cost 4.6, 16 missed frauds against 118), so the learned model adds real but modest value.
- A looser rule (amount greater than or equal to balance) fails completely, because 81% of legitimate transactions satisfy it in this dataset.
- Feature ablation shows the score depends almost entirely on one simulator pattern: the amount equals the sender's balance. The pattern never occurs among legitimate PaySim transactions, so it will not transfer to real fraud.
- A first experiment that used post-transaction balances scored higher still and was discarded as unrealistic, because those balances are unknown before approval.
- Cost sensitivity (missed-fraud cost 10 to 500 with the false-alert cost fixed at 5, so ratios from 2:1 to 100:1): gradient boosting has the lowest cost at every ratio, so the model ranking does not depend on the cost assumption. The size of its advantage does: its cost is 17% lower than the rule's at 2:1 (3.5 against 4.2 per 1k), 63% lower at 20:1 (4.6 against 12.5) and 81% lower at 100:1 (9.6 against 49.6). If misses are cheap, the rule alone gives up very little.
- Gradient boosting flags the same transactions at every ratio (3.33 alerts per 1k, 16 missed frauds) because PaySim fraud is almost perfectly separable. Real data would move the threshold far more. Logistic regression shows the expected behaviour: alerts rise from 1.89 to 7.61 per 1k and misses fall from 1,963 to 378 as a miss gets costlier. The looser rule (amount greater than or equal to balance) never flags anything at any ratio.

**Product implication:** the API returns both the model decision and the rule flag, so a consumer can see when they agree and when they do not. If real costs make the extra 8 cost units per 1k (rule 12.5 against model 4.6) not worth the added complexity, shipping the rule alone would be a legitimate decision.

## 7. Scope and milestones

See [ROADMAP.md](ROADMAP.md). Done: data layer, baseline, model comparison, ablation, scoring API, Docker. Planned: drift monitoring (M5), LLM assistant (M6), final write-up (M7).

## 8. Risks and open questions

| Risk or question | Why it matters | Planned handling |
|---|---|---|
| Synthetic data | Real fraud adapts and is messier | State the limitation clearly; do not claim transfer |
| Cost assumptions | Threshold and ranking of models depend on them | Done: sweep from 2:1 to 100:1. Ranking is stable on PaySim, but see the separability caveat in section 6 |
| Scores are not calibrated probabilities | Cannot be read as "probability of fraud" | Calibrate on validation data before any probability use |
| Label delay and drift | Fraud labels arrive late and patterns change | Drift monitoring (M5); retraining trigger |
| Fairness and explainability | A regulated setting needs both | Not analysed yet; add slice-level metrics and reason codes |
| Single run | No uncertainty on the reported numbers | Bootstrap confidence intervals |
| Pickled model bundle | Loading untrusted pickles is unsafe | Load only self-built bundles; consider a safer format |

## 9. Rollout plan (if this were a real product)

1. **Shadow mode:** score live traffic without acting, and compare flags against later fraud labels.
2. **Threshold review:** with real costs, re-tune the threshold and check calibration.
3. **Limited rollout:** send flagged transactions to a human review queue for a small share of traffic, with an A/B holdout.
4. **Guardrails:** false-alert rate, customer-friction signals and fraud loss per 1k transactions, with explicit rollback criteria if any guardrail degrades.
5. **Monitoring:** feature and score drift, alert volume per day, and periodic recalibration.
