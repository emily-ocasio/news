DROP VIEW articles_type;
CREATE VIEW articles_type AS
SELECT a.*, SUM(IIF(t.TypeID IN (7,8,9,10,13,19,21), 0, 1)) AS GoodTypes
FROM articles a 
JOIN articletypes t
ON a.RecordId = t.RecordId
GROUP BY a.RecordId