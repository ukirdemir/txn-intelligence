-- How often does a transaction drain the origin account, split by fraud label?
SELECT
    is_fraud,
    COUNT(*) AS n_txns,
    ROUND(
        100.0 * SUM(CASE WHEN new_balance_orig = 0 AND old_balance_orig > 0 THEN 1 ELSE 0 END)
        / COUNT(*),
        3
    ) AS pct_origin_emptied
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;
