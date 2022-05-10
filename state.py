"""
Type Definitions, including:
    State NamedTuple
    and for type hinting Action, Reaction, RxResp
"""
from typing import NamedTuple, Tuple, Optional, Any, Callable

class State(NamedTuple):
    """
    Contains all state for application
    """
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
RxResp = Tuple[Action, State]

# A reaction (no side effects) takes a State
#   and returns a RxResp (see above)
Reaction = Callable[[State], RxResp]
