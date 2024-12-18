SELECT a.RecordId, SUM(1 << t.TypeId) BinaryTypes
FROM articles a
JOIN articletypes t ON a.RecordId = t.RecordId
GROUP BY t.RecordId
LIMIT 20