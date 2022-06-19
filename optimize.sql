EXPLAIN QUERY PLAN
        SELECT 
            a.*
        FROM articles a 
        WHERE a.Dataset = "CLASS"
        AND a.Status IS NULL
        AND a.Autoclass = "M"
        AND a.PubDate IN (
            SELECT PubDate
            FROM dates
            WHERE PubDate IN (
                SELECT DISTINCT PubDate
                FROM articles
                WHERE Dataset = "CLASS"
                AND Status IS NULL
                AND Autoclass = "M"
            )
            ORDER BY Priority
            LIMIT ?
        )
        ORDER BY a.PubDate, a.RecordId
        
