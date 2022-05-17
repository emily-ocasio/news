SELECT COUNT(*)
FROM articles
WHERE Dataset = "CLASS"
AND Status IS NULL
AND AutoClass = "M"
AND PubDate IN (
    SELECT PubDate 
    FROM dates
    WHERE PubDate IN (
        SELECT DISTINCT PubDate
        FROM articles
        WHERE Dataset = "CLASS"
        AND Status IS NULL
    )
    ORDER BY Priority
    LIMIT 25


)
