"""
Type Definitions, including:
    State NamedTuple
    and for type hinting Action, Reaction, RxResp
"""
from collections.abc import Callable
from typing import NamedTuple, Optional, Any, List
from enum import Enum
from sqlite3 import Row
from immutabledict import immutabledict
from pydantic import BaseModel, Field

Rows = tuple[Row,...]
TextChoice = immutabledict[str, str]
class HomicideClass(str, Enum):
    """
    Enum for homicide classification
    """
    HOMICIDE = "homicide"
    VEHICULAR_HOMICIDE = "vehicular homicide"
    KILLED_BY_LAW_ENFORCEMENT = "killed by law enforcement"
    FICTIONAL_HOMICIDE = "fictional homicide"
    NO_HOMICIDE_IN_ARTICLE = "no homicide in article"


class LocationClass(str, Enum):
    """
    Enum for location classification
    """
    NOT_IN_MASSACHUSETTS = "no homicide in Massachusetts"
    IN_MASSACHUSETTS = "homicide(s) in Massachusetts"


class County(str, Enum):
    """
    Enum for county where the crime occurred
    """
    BARNSTABLE = "Barnstable"
    BERKSHIRE = "Berkshire"
    BRISTOL = "Bristol"
    DUKES = "Dukes"
    ESSEX = "Essex"
    FRANKLIN = "Franklin"
    HAMPDEN = "Hampden"
    HAMPSHIRE = "Hampshire"
    MIDDLESEX = "Middlesex"
    NANTUCKET = "Nantucket"
    NORFOLK = "Norfolk"
    PLYMOUTH = "Plymouth"
    SUFFOLK = "Suffolk"
    WORCESTER = "Worcester"
    NOT_IN_MASSACHUSETTS = "not in Massachusetts"

class Relationship(str, Enum):
    """
    Relationship between the victim and the offender
    From the point of view of the victim
    """
    ACQUAINTANCE = "acquaintance"
    SON = "son"
    DAUGHTER = "daughter"
    HUSBAND = "husband"
    STRANGER = "stranger"
    WIFE = "wife"
    BROTHER = "brother"
    SISTER = "sister"
    OTHER_FAMILY = "other family"
    GIRLFRIEND = "girlfriend"
    BOYFRIEND = "boyfriend"
    NEIGHBOR = "neighbor"
    STEPFATHER = "stepfather"
    STEPMOTHER = "stepmother"
    STEPSON = "stepson"
    FRIEND = "friend"
    OTHER_KNOWN_TO_VICTIM = "other known to victim"
    MOTHER = "mother"
    FATHER = "father"
    IN_LAW = "in-law"
    EMPLOYEE = "employee"
    HOMOSEXUAL_RELATIONSHIP = "homosexual relationship"


class Sex(str, Enum):
    """ Enum for sex """
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class Weapon(str, Enum):
    """ Enum for weapon """
    SHOTGUN = "shotgun"
    RIFLE = "rifle"
    HANDGUN = "handgun, pistol, revolver, etc"
    FIREARM = "other firearm"
    KNIFE = "knife or cutting instrument"
    BLUNT_OBJECT = "blunt object - hammer, club, etc"
    BEATING = "personal weapon, including beating"
    FIRE = "fire, arson"
    STRANGULATION = "strangulation or hanging"
    ASPHYXIATION = "asphyxiation"
    DRUGS = "sleeping pills or other drugs"
    EXPLOSIVES = "explosives"
    DROWNING = "drowning"
    OTHER = "other"
    UNKNOWN = "unknown"


class Circumstance(str, Enum):
    """
    Enum for circumstances of the crime
    """
    NO_KILLING = "victim is not dead may be in critical condition"
    ARSON = "arson"
    VEHICULAR_HOMICIDE = "vehicular homicide"
    FELON_KILLED_BY_POLICE = "death of felon at hands of police"
    INSTITUTIONAL_KILLING = \
        "killing within an institution such as jail or mental hospital"
    BURGLARY = "burglary"
    ROBBERY = "robbery"
    BRAWL = "brawl"
    RAPE = "rape or other sex offense"
    ARGUMENT = "argument"
    GANG_KILLING = "gang-related killing"
    LOVERS_TRIANGLE = "lover's triangle"
    CHILD_KILLED_BY_BABYSITTER = "child killed by babysitter"
    NEGLIGENCE = "negligence"
    OTHER = "all other"
    UNDETERMINED = "undetermined"


class LocationClassResponse(BaseModel):
    """
    Response model for location classification
    """
    classification: LocationClass


class HomicideClassResponse(BaseModel):
    """
    Response model for homicide classification.
    """
    classification: HomicideClass


class Victim(BaseModel):
    """
    Model for victim information
    """
    victim_name: str | None = Field(description = "Name of victim")
    victim_age: int | None = Field(description = "Age of victim")
    victim_sex: Sex | None = Field(description = "Sex of victim")
    suspect_count: int | None = Field(description = "Number of suspects")
    suspect_age: int | None = Field(description = "Age of main suspect")
    suspect_sex: Sex | None = Field(description = "Sex of main suspect")
    date_of_death: str | None = Field(
        description="Approximate date that victim was found dead (YYYY-MM-DD)")
    weapon: Weapon | None = Field(
        description = "Type of weapon used to kill victim")
    relationship: Relationship | None = Field(
        description="The victim's relationship to the killer")
    circumstance: Circumstance | None = Field(
        description="Circumstances of the crime")
    county: County = Field(
        description="County where the homicide occurred (infer from other "
            "details if not explicitly stated). Boston is in Suffolk county.")
    town: str | None = Field(
        description="Town or city where the homicide occurred")
    victim_details: str = Field(
        description = "All the text in the article that refers to this "
            "victim, including background on the victim, circumstances of "
            "the crime, and any investigation or trial. "
            "Include every detail mentioned in the article about the victim.")

class ArticleAnalysisResponse(BaseModel):
    """
    Response model for extracting information about victims
    """
    homicide_victims: List[Victim] = Field(
            description="List of homicide victims in the article. "
                "Only include dead victims.")

class State(NamedTuple):
    """
    Contains all state for application
    """
    next_event: str
    end_program: bool = False
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
    assign_choice: str = ''
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
    gpt_model: str = 'mini'
    gpt_max_tokens: int = 256
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
    #current_step: str = 'not_started'
    next_step: str = 'begin'


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
