-- Which balance relationship separates fraud?
-- Among TRANSFER / CASH_OUT with a positive origin balance, count how often the amount is
-- greater than, equal to, or at least the sender's old balance, split by fraud label.
SELECT
    is_fraud,
    COUNT(*) AS n_txns,
    SUM(CASE WHEN amount >= old_balance_orig THEN 1 ELSE 0 END) AS n_amount_ge_balance,
    SUM(CASE WHEN amount = old_balance_orig THEN 1 ELSE 0 END) AS n_amount_eq_balance,
    SUM(CASE WHEN amount > old_balance_orig THEN 1 ELSE 0 END) AS n_amount_gt_balance
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT') AND old_balance_orig > 0
GROUP BY is_fraud
ORDER BY is_fraud;
