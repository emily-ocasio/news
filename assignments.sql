CREATE VIEW assignments AS
    SELECT
        s.*,
        COUNT(t.RecordId) AssignCount
    FROM shr s
    LEFT JOIN topics t ON s."index" = t.ShrId
    GROUP BY s."index"