"""
Type Definitions, including:
    State NamedTuple
    and for type hinting Action, Reaction, RxResp
"""
from collections.abc import Callable
from typing import NamedTuple, Optional, Any

class State(NamedTuple):
    """
    Contains all state for application
    """
    next_event: str
    article_date: Optional[str] = None
    article_id: Optional[int] = None
    articles: tuple = tuple()
    matches: tuple = tuple()
    nomatches: tuple = tuple()
    FP: tuple = tuple()
    FN: tuple = tuple()
    TP: tuple = tuple()
    TN: tuple = tuple()
    article_kind: Optional[str] = None
    review_dataset: Optional[str] = None
    review_label: Optional[str] = None
    article_lines: tuple = tuple()
    remaining_lines: bool = False
    current_article_types: tuple = tuple()
    new_label: str = ''
    next_article: int = 0
    dates_to_classify: int = 0
    dates_to_reclassify: int = 0
    dates_to_assign: int = 0
    choice_type: str = ''
    query_type: str = ''
    outputs: Any = None
    inputargs: Any = tuple()
    inputkwargs: Any = {}

# An Action is a function that has State as argument,
#   performs a side effect, and returns an updated State
Action = Callable[[State], State]

# The response of a reaction (RxResp) is a tuple,
#   which includes the next Action to take, and the updated State
#RxResp = Tuple[Callable[...,State], State]
RxResp = tuple[Action, State]

# A reaction (no side effects) takes a State
#   and returns a RxResp (see above)
Reaction = Callable[[State], RxResp]
