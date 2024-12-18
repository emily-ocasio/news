"""
Type Definitions, including:
    State NamedTuple
    and for type hinting Action, Reaction, RxResp
"""
from collections.abc import Callable
from typing import NamedTuple, Optional, Any
from enum import Enum
from sqlite3 import Row
from pydantic import BaseModel

Rows = tuple[Row,...]
class HomicideClass(Enum):
    """
    Enum for homicide classification
    """
    HOMICIDE = "homicide"
    VEHICULAR_HOMICIDE = "vehicular homicide"
    KILLED_BY_LAW_ENFORCEMENT = "killed by law enforcement"
    NO_HOMICIDE_IN_ARTICLE = "no homicide in article"

class HomicideClassResponse(BaseModel):
    """
    Response model for homicide classification.
    """
    classification: HomicideClass

class State(NamedTuple):
    """
    Contains all state for application
    """
    next_event: str
    user: str = ''
    terminal_size: tuple[int, int] = 0, 0
    main_flow: str = 'start'
    review_type: str = ''
    last_flow: str = ''
    article_date: Optional[str] = None
    article_id: Optional[int] = None
    multi_count: int = 1
    multi_mod: int = 0
    articles: Rows = tuple()
    matches: Rows = tuple()
    nomatches: Rows = tuple()
    homicides: Rows = tuple()
    homicides_retrieved: bool = False
    current_homicide: int = -1
    homicide_action: str = ''
    homicides_assigned: Rows = tuple()
    homicide_group: str = ''
    selected_homicide: int = 0
    selected_homicides: tuple[int,...] = tuple()
    victim: str = ''
    county: str = ''
    FP: Rows = tuple()
    FN: Rows = tuple()
    TP: Rows = tuple()
    TN: Rows = tuple()
    article_kind: Optional[str] = None
    review_dataset: str = ''
    articles_retrieved: bool = False
    refresh_article: bool = False
    review_label: Optional[str] = None
    article_lines: tuple = tuple()
    remaining_lines: bool = False
    current_article_types: Rows = tuple()
    new_label: str = ''
    new_notes: str = ''
    next_article: int = 0
    dates_to_classify: int = 0
    dates_to_reclassify: int = 0
    reclassify_begin: str = '1976-01'
    reclassify_end: str = '1984-12'
    dates_to_assign: int = 0
    assign_begin: str = '1976-01'
    assign_end: str = '1984-12'
    gpt3_action: str = 'humanize'
    gpt3_source: str = 'article'
    pre_article_prompt: str = 'reporter'
    post_article_prompt: str = '3L_not3'
    humanizing: str = ''
    humanizing_saved: bool = False
    extract: str = ''
    gpt3_prompt: str = ''
    gpt3_response: str = ''
    homicide_month: str = ''
    homicide_victim: str = ''
    choice_type: str = ''
    query_type: str = ''
    outputs: Any = None
    inputargs: Any = tuple()
    inputkwargs: Any = {}
    articles_to_filter: int = 0
    homicide_filter_status: str = ''

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
