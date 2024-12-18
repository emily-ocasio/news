UPDATE articles
SET Status = NULL, Dataset = "VAL2"
WHERE Dataset = "CLASS"
AND PubDate in (
    SELECT PubDate
    FROM dates
    WHERE Priority > 1  AND Priority < 42
)

