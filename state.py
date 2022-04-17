from typing import NamedTuple, Tuple, Optional, Any
from sqlite3 import Connection

class State(NamedTuple):
    next_event: str
    article_date: Optional[str] = None
    article_id: Optional[int] = None
    articles: Tuple = tuple()
    matches: Tuple = tuple()
    nomatches: Tuple = tuple()
    FP: Tuple = tuple()
    FN: Tuple = tuple()
    TP: Tuple = tuple()
    TN: Tuple = tuple()
    article_kind: Optional[str] = None
    review_dataset: Optional[str] = None
    review_label: Optional[str] = None
    article_lines: Tuple = tuple()
    remaining_lines: bool = False
    current_article_types: Tuple = tuple()
    new_label: str = ''
    next_article: int = 0
    dates_to_classify: int = 0
    outputs: Any = None
    inputargs: Any = tuple()
    inputkwargs: Any = {}
