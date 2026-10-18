-- Daily transaction volume and fraud rate (step = hour index).
SELECT
    step // 24 AS day,
    COUNT(*) AS n_txns,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY day
ORDER BY day;
