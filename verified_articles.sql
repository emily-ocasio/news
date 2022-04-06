DROP VIEW verified_articles;
CREATE VIEW verified_articles AS
SELECT a.*, v.Status, v.dataset, SUM(IIF(t.TypeID IN (7,8,9,10,13,19,21), 0, 1)) AS GoodTypes
FROM articles a 
JOIN verifications v ON a.RecordId = v.RecordId
JOIN articletypes t ON a.RecordId = t.RecordId
GROUP BY t.RecordId