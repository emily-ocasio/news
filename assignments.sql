CREATE VIEW assignments AS
    SELECT
        s.*,
        COUNT(t.RecordId) AssignCount,
        MAX(IIF(t.Human = 3,1,IIF(t.Human>0,0,NULL))) AS Humanized
    FROM shr s
    LEFT JOIN topics t ON s."index" = t.ShrId
    GROUP BY s."index"
