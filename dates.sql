select dataset, count(distinct pubdate)
from verified_articles
where dataset is not null
group by dataset
