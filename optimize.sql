EXPLAIN QUERY PLAN
        SELECT 
            a.*
        FROM articles a 
        WHERE a.Dataset = "CLASS"
        AND a.Status = 'M'
        AND a.AssignStatus IS NULL
        AND a.PubDate IN (
            SELECT PubDate
            FROM dates
            WHERE PubDate IN (
                SELECT DISTINCT PubDate
                FROM articles
                WHERE Dataset = "CLASS"
                AND Status = 'M'
                AND AssignStatus IS NULL
            )
            ORDER BY Priority
            LIMIT ?
        )
        ORDER BY a.PubDate, a.RecordId
        
