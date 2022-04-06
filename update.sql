UPDATE articles
SET Status = (SELECT Status FROM verifications WHERE verifications.RecordId = articles.RecordId),
Dataset = (SELECT Dataset FROM verifications WHERE verifications.RecordId = articles.RecordId)
WHERE articles.RecordId IN (SELECT verifications.RecordId from verifications)