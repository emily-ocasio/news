WITH orders AS MATERIALIZED (
    SELECT ROW_NUMBER() OVER (
        ORDER BY RANDOM()
    ) rownum,
    PUBDATE
    FROM dates
    WHERE Priority = 0
)
UPDATE dates
SET Priority = (SELECT rownum+1 FROM orders where orders.pubdate = dates.pubdate)
WHERE Priority = 0