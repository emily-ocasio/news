SELECT t.Human, t.HumanManual, count(*)
FROM topics t
WHERE t.Human IS NOT NULL
    AND t.RecordId IN
    (
        SELECT a.RecordId
        FROM articles a
        WHERE a.Dataset = 'VAL'
    )
GROUP BY t.Human, t.HumanManual
;