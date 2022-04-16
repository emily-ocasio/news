SELECT * FROM (
    SELECT a.*, SUM(IIF(t.TypeId IN (7,8,9,10,13,19,21), 0, 1)) as GoodType
    FROM articles a 
    JOIN articletypes t 
    ON a.RecordID = t.RecordId
    WHERE Pubdate = "19840331"
    GROUP BY a.RecordId
    HAVING GoodType >= 0
)
