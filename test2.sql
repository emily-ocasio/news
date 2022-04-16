UPDATE articles
SET Dataset = "TEST2"
WHERE Dataset = "VAL2"
AND PubDate IN (
    SELECT PubDate
    FROM dates
    WHERE Priority >=22 AND Priority <= 41
)