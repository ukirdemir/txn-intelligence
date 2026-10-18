# Product Requirements Document: Transaction Intelligence

| | |
|---|---|
| Owner | Umut Kirdemir |
| Status | Draft |
| Last updated | TODO |

## 1. Problem

TODO: what goes wrong today? (fraud losses, false declines that frustrate customers, manual review load.) Quantify with assumed numbers and state the assumptions.

## 2. Users and jobs to be done

| User | Job to be done | Pain today |
|---|---|---|
| Fraud analyst | Prioritise suspicious transactions for review | Too many false alerts |
| Customer | Understand and trust spending summaries | Opaque categories |
| Product owner | Tune the risk/friction trade-off | No cost visibility |

## 3. Goals and non-goals

**Goals**
- Detect fraud with a measurable cost reduction versus a simple baseline.
- Expose a scoring API with a documented latency budget.
- Provide natural-language spending insights.

**Non-goals**
- Production-grade real-time feature store.
- Use of any real customer data.

## 4. Success metrics

| Metric | Target | How measured |
|---|---|---|
| PR-AUC | TODO vs baseline | Held-out time-based split |
| Expected cost per 1k transactions | TODO % lower than baseline | Cost matrix in `evaluate.py` |
| API p95 latency | TODO ms | Load test |
| Assistant answer accuracy | TODO % | Eval set of questions with SQL ground truth |

## 5. Decision framework

Cost matrix (assumed, to be justified): cost of a missed fraud (FN), cost of a false alert (FP), cost of manual review. The decision threshold is chosen by minimising expected cost, not by the default 0.5.

## 6. Scope and milestones

See [ROADMAP.md](ROADMAP.md).

## 7. Risks and open questions

- Label delay and concept drift.
- Synthetic data may not reflect real fraud patterns.
- Fairness and explainability requirements for a regulated setting.

## 8. Experiment and rollout plan

TODO: shadow mode, then limited rollout, with guardrail metrics and rollback criteria.
