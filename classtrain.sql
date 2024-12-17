-- Create a CTE to store the sample counts
WITH stratified_counts AS (
    SELECT
        status,
        SUBSTR(pubdate, 1, 4) AS year,
        COUNT(*) AS total_count
    FROM articles
    WHERE dataset = 'CLASS' AND autoclass = 'M'
    GROUP BY status, year
),
numbered_articles AS (
    SELECT
        a.status,
        SUBSTR(a.pubdate, 1, 4) AS year,
        a.RecordId,
        ROW_NUMBER() OVER (PARTITION BY a.status, SUBSTR(a.pubdate, 1, 4) ORDER BY RANDOM()) AS row_num
    FROM articles a
    JOIN stratified_counts sc
    ON a.status = sc.status AND SUBSTR(a.pubdate, 1, 4) = sc.year
    WHERE a.dataset = 'CLASS' AND a.autoclass = 'M'
),
random_samples AS (
    SELECT
        status,
        year,
        RecordId
    FROM numbered_articles
    WHERE row_num <= 20
)
-- Update the dataset for the selected random samples
UPDATE articles
SET dataset = 'CLASSTRAIN'
WHERE RecordId IN (SELECT RecordId FROM random_samples);