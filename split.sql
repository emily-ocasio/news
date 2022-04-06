update verifications
set dataset="TRAIN"
where recordid in
(
select v.recordid
from verifications v 
join articles a 
on v.recordid = a.recordid
where a.pubdate 
in
(
select distinct pubdate
from articles a
join verifications v
on a.recordid = v.recordid
where status="M"
order by random()
limit 25
)
and v.status = "M"
)
;
