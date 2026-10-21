# Model Card

## Model details
- Type: TODO (e.g. PyTorch MLP on tabular features)
- Version: 0.1.0
- Owner: Umut Kirdemir

## Intended use
TODO: portfolio demonstration of fraud scoring. Not for production decisions about real customers.

## Data
- Dataset: PaySim (synthetic mobile-money transactions), from Kaggle. TODO: confirm the licence on the dataset page and record it here.
- Split: time-ordered 60% train / 20% validation / 20% test by `step`. Thresholds are tuned on validation only.
- Test period: 1,272,524 transactions, 4,254 frauds (0.33%). Fraud only occurs in TRANSFER and CASH_OUT.

## Metrics
Test-period results with pre-approval features (see README for the table): gradient boosting PR-AUC 0.999, MLP 0.970, logistic regression 0.817. Cost per 1k transactions: 4.6, 28.3 and 77.4, against 334.3 if nothing is flagged. A one-line rule (TRANSFER/CASH_OUT where the amount exactly equals the sender's balance) reaches PR-AUC 0.972 and cost 12.5, ahead of the MLP and logistic regression but behind gradient boosting (16 missed frauds against 118 for the rule).

With post-transaction balances included, MLP and gradient boosting reached PR-AUC 1.000. That setting is unrealistic (balances are not known before approval) and is kept only as a comparison.

Feature ablation (gradient boosting, PR-AUC): all features 0.999; without destination features 1.000; without transaction type 0.999; without the explicit account-emptying features 0.922; without any origin-balance information 0.526; amount, hour and type only 0.263. Permutation importance is concentrated in `orig_would_be_emptied` and `amount_to_orig_balance`.

TODO: bootstrap confidence intervals.

## Evaluation by slice
TODO: performance by amount band, transaction type, and other slices.

## Limitations and risks
- Synthetic data: PaySim fraud follows a simple script, so scores overstate real-world performance.
- Performance depends almost entirely on one simulator pattern (the transaction amount exactly equals the sender's balance, which never happens for legitimate transactions in this dataset); see the feature ablation and the rule baseline.
- Fraud rate differs between periods (test period 0.33%) and between days (daily fraud share from 0.05% to over 4%, because legitimate volume collapses on some days while fraud keeps arriving). Drift is monitored in batch (`monitor.py`); nothing retrains or adapts automatically.
- Cost values are assumptions, and the chosen threshold depends on them.
- No fairness, explainability or label-delay analysis yet.
- Single dataset, single split, no confidence intervals.
- Scores are not calibrated probabilities (class-weighted training). Calibrate on the validation period (for example isotonic regression) before using them as probabilities.

## Maintenance
Monitoring: run `monitor.py` daily against a reference period; read score PSI and alerts per 1,000 first, then input PSI. Suggested review triggers (assumed, to be tuned on real traffic): score PSI above 0.25 or alerts per 1,000 more than 3 times the reference. TODO: retraining trigger, and delayed-label recall once labels arrive.
