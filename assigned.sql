UPDATE assigned
SET GroupPriority = randomized.k
FROM (
    SELECT ROW_NUMBER() OVER (ORDER BY RANDOM()) as k, ShrId
    FROM assigned a2 
    WHERE a2.ArticleCount = 14
) AS randomized
WHERE assigned.ShrId = randomized.ShrId