WITH datelist AS MATERIALIZED (
    SELECT DISTINCT PubDate
    FROM articles
    INDEXED BY Dataset
    WHERE Dataset = "CLASS"
    AND Pubdate NOT IN 
    (
        SELECT DISTINCT PubDate
        FROM articles
        INDEXED BY Dataset
        WHERE Dataset != "CLASS"
    )
)
UPDATE dates
SET Complete = 1
WHERE PubDate IN (SELECT PubDate FROM datelist)
