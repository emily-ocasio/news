DELETE FROM verifications WHERE status = "P" and RecordId IN (
	SELECT v1.recordid
	FROM verifications v1
	JOIN verifications v2
	ON v1.recordid = v2.recordid
	where v1.status = "P" and v2.status<>"P"
)
;
