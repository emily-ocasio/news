SELECT
    s."index",
    COALESCE(t.cnt, 0) as topic_count,
    COUNT(*) as shr_count
FROM shr s
LEFT JOIN (
    SELECT shrid, COUNT(*) as cnt
    FROM topics
    GROUP BY shrid
) t ON s."index" = t.shrid
where s.state = 'Massachusetts'
GROUP BY topic_count 
ORDER BY topic_count
;