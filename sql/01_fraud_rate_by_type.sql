-- Fraud volume and rate by transaction type.
-- Table: transactions (see txn_intelligence.data.PAYSIM_RENAME for columns).
SELECT
    type,
    COUNT(*) AS n_txns,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY type
ORDER BY n_fraud DESC;
