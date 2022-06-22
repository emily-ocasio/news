-- SQLite
SELECT a.RecordId, Title, Notes, count(t.shrid) as cnt
FROM articles a
LEFT OUTER JOIN topics t 
ON a.RecordId = t.RecordId
WHERE AssignStatus = 'D'
GROUP BY a.RecordId
HAVING cnt = 0