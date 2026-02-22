"""
Type Definitions, including:
    State NamedTuple
    and for type hinting Action, Reaction, RxResp
"""
from collections.abc import Callable
from typing import NamedTuple, Optional, Any, List
from enum import Enum
import json
from sqlite3 import Row
from immutabledict import immutabledict
from pydantic import BaseModel, Field, ConfigDict

Rows = tuple[Row, ...]
TextChoice = immutabledict[str, str]


class HomicideClass(str, Enum):
    """
    Enum for homicide classification
    """
    VEHICULAR_HOMICIDE = "vehicular homicide"
    KILLED_BY_LAW_ENFORCEMENT = "killed by law enforcement"
    FICTIONAL_HOMICIDE = "fictional homicide"
    MILITARY_KILLINGS = "military killings"
    NO_HOMICIDE_IN_ARTICLE = "no homicide in article"
    OTHER_ACTUAL_HOMICIDE = "other actual homicide"

class Article_Classification(str, Enum): # pylint:disable=invalid-name
    """
    Enum for article classification
    """
    NO_HOMICIDE_IN_ARTICLE = "no homicide in article"
    HOMICIDE_BEFORE_1977 = "homicide before 1977"
    HOMICIDES_OUTSIDE_WASHINGTON_DC = "homicides outside Washington, D.C."
    HOMICIDE_IN_WASHINGTON_DC_SINCE_1977 = \
        "homicide in Washington, D.C. since 1977"

class Homicide_Classification(str, Enum): # pylint:disable=invalid-name
    """
    Enum for homicide classification
    """
    VEHICULAR_HOMICIDE = "vehicular homicide"
    FICTIONAL_HOMICIDE = "fictional homicide"
    MILITARY_KILLINGS = "military killings"
    OTHER_ACTUAL_HOMICIDE = "other actual homicide"

class LocationClass(Enum):
    """
    Enum for location classification
    """
    NOT_IN_MASSACHUSETTS = "no homicide in Massachusetts"
    IN_MASSACHUSETTS = "homicide(s) in Massachusetts"


class LocationClassDC(str, Enum):
    """
    Enum for location classification for DC
    """
    NOT_IN_DC = "not in Washington, D.C."
    IN_DC = "in Washington, D.C."

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
    EX_WIFE = "ex-wife"
    EX_HUSBAND = "ex-husband"
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
    EMPLOYER = "employer"
    HOMOSEXUAL_RELATIONSHIP = "homosexual relationship"


class Sex(str, Enum):
    """ Enum for sex """
    MALE = "male"
    FEMALE = "female"

class Race(str, Enum):
    """ Enum for race """
    WHITE = "White"
    BLACK = "Black"
    ASIAN = "Asian"
    NATIVE_AMERICAN = "Native American"

class Ethnicity(str, Enum):
    """ Enum for ethnicity """
    HISPANIC = "Hispanic"
    NON_HISPANIC = "Non-Hispanic"

class KillingMethod(str, Enum):
    """ Enum for weapon """
    SHOTGUN = "shotgun"
    RIFLE = "rifle"
    HANDGUN = "handgun"
    FIREARM = "firearm"
    KNIFE = "knife"
    BLUNT_OBJECT = "blunt object"
    BEATING = "beating"
    FIRE = "fire"
    STRANGULATION = "strangulation"
    ASPHYXIATION = "asphyxiation"
    DRUGS = "drugs"
    EXPLOSIVES = "explosives"
    DROWNING = "drowning"
    POISON = "poison"
    PUSHED_FROM_HEIGHT = "pushed from height"
    OTHER = "other"
    UNKNOWN = "unknown"


class Circumstance(str, Enum):
    """
    Enum for circumstances of the crime
    """
    #NO_KILLING = "victim is not dead may be in critical condition"
    ARSON = "arson"
    VEHICULAR_HOMICIDE = "vehicular homicide"
    FELON_KILLED_BY_POLICE = "felon killed by police"
    INSTITUTIONAL_KILLING = "institutional killing"
    BURGLARY = "burglary"
    ROBBERY = "robbery"
    BRAWL = "brawl"
    RAPE = "rape"
    ARGUMENT = "argument"
    GANG_KILLING = "gang killing"
    LOVERS_TRIANGLE = "lover's triangle"
    CHILD_KILLED_BY_BABYSITTER = "child killed by babysitter"
    NEGLIGENCE = "negligence"
    NARCOTICS_RELATED = "narcotics related"
    OTHER_FELONY_RELATED = "other felony related"
    OTHER = "other"
    UNDETERMINED = "undetermined"


class LocationClassResponse(BaseModel):
    """
    Response model for location classification
    """
    classification: LocationClass


class LocationClassDCResponse(BaseModel):
    """
    Response model for location classification for DC
    """
    classification: LocationClassDC


class HomicideClassResponse(BaseModel):
    """
    Response model for homicide classification.
    """
    classification: HomicideClass

class HomicideClassDCResponse(BaseModel):
    """
    Response model for homicide classification for DC.
    """
    classification: HomicideClass
    location: LocationClassDC | None



class VictimBase(BaseModel):
    """
    Base model for victim information
    """
    victim_name: str | None = Field(description="Name of victim")
    victim_age: int | None = Field(description="Age of victim")
    victim_sex: Sex | None = Field(description="Sex of victim")
    victim_count: int | None = Field(
        description="Number of victims murdered in the same incident",
        default=None)
    suspect_count: int | None = Field(description="Number of suspects",
        default=None)
    suspect_age: int | None = Field(description="Age of main suspect at "
                                    "the time of the crime",
                                    default=None)
    suspect_sex: Sex | None = Field(description="Sex of main suspect",
                                    default=None)
    date_of_death: str | None = Field(
        description="Approximate date that victim was found dead (YYYY-MM-DD)")
    killing_method: KillingMethod | None = Field(
        description="Type of weapon used to kill victim",
        default=None)
    relationship: Relationship | None = Field(
        description="The victim's relationship to the killer",
        default=None)
    circumstance: Circumstance | None = Field(
        description="Circumstances of the crime",
        default=None)
    victim_details: str = Field(
        description="All the text in the article that refers to this "
        "victim, including background on the victim, circumstances of "
        "the crime, and any investigation or trial. "
        "Include every detail mentioned in the article about the victim.")

class Victim(BaseModel):
    """
    Model for Victim information
    """
    model_config = ConfigDict(extra="forbid")
    victim_name: str | None
    victim_age: int | None
    victim_sex: Sex | None
    victim_race: Race | None
    victim_ethnicity: Ethnicity | None
    relationship: Relationship | None
class Incident(BaseModel):
    """
    Model for Incident information
    """
    model_config = ConfigDict(extra="forbid")
    year: int
    month: int| None
    day: int | None
    location: str | None
    circumstance: Circumstance
    killing_method: KillingMethod
    offender_count: int | None
    offender_name: str | None
    offender_age: int | None
    offender_sex: Sex | None
    offender_race: Race | None
    offender_ethnicity: Ethnicity | None
    victim_count: int | None
    summary: str | None
    victim: list[Victim] = Field(min_length=1)

class WashingtonPostArticleAnalysis(BaseModel):
    """
    Response model for Washington Post article analysis.
    """
    model_config = ConfigDict(extra="forbid")
    article_classification: Article_Classification
    homicide_classification: Homicide_Classification | None
    incidents: list[Incident]

    @property
    def incidents_json(self) -> str:
        """
        Return the incidents as a JSON string."""
        if len(self.incidents) == 0:
            return ""
        incidents_list = self.model_dump()['incidents']
        return json.dumps(incidents_list, indent=2)

    @property
    def result_str(self) -> str:
        """
        Return a string summary of the classification results.
        """
        art = self.article_classification.value
        hom = self.homicide_classification.value \
            if self.homicide_classification else 'None'
        return f"GPT article classification: {art}\n" + \
            f"GPT homicide classification: {hom}\n" + \
            self.incidents_json

class WashingtonPostArticleHomicideClassification(BaseModel):
    """
    Response model for Washington Post article homicide classification
    Does not include incidents.
    """
    model_config = ConfigDict(extra="forbid")
    article_classification: Article_Classification
    homicide_classification: Homicide_Classification | None

    @property
    def result_str(self) -> str:
        """
        Return a string summary of the classification results.
        """
        art = self.article_classification.value
        hom = self.homicide_classification.value \
            if self.homicide_classification else 'None'
        return f"GPT article classification: {art}\n" + \
            f"GPT homicide classification: {hom}\n" + \
            self.model_dump_json(indent=2)

class WashingtonPostArticleIncidentExtraction(BaseModel):
    """
    Response model for Washington Post article extraction of incident details.
    """
    incidents: list[Incident]

    @property
    def incidents_json(self) -> str:
        """
        Return the incidents as a JSON string."""
        if len(self.incidents) == 0:
            return ""
        incidents_list = self.model_dump()['incidents']
        return json.dumps(incidents_list, indent=2)

    @property
    def result_str(self) -> str:
        """
        Return a string summary of the results (incidents only).
        """
        return self.incidents_json

class VictimDC(VictimBase):
    """
    Model for Victim information in Washington, DC
    """
    location: LocationClassDC = Field(
        description="Location of the homicide")


class VictimMass(VictimBase):
    """Model for Victim information in Massachusetts"""
    county: County = Field(
        description="County where the homicide occurred (infer from other "
        "details if not explicitly stated). Boston is in Suffolk county.")
    town: str | None = Field(
        description="Town or city where the homicide occurred")


class ArticleAnalysisResponse(BaseModel):
    """
    Response model for extracting information about victims
    """
    homicide_victims: List[Victim] = Field(
        description="Information for each homicide victim in the article. "
        "Only include dead victims.")


class ArticleAnalysisDCResponse(BaseModel):
    """
    Response model for extracting information about victims
    """
    homicide_victims: List[VictimDC] = Field(
        description="Information for each homicide victim in the article. "
        "Only include dead victims.")


class GenericVictim(VictimBase):
    """
    Generic model for Victim information
    """
    location: LocationClass | None = Field(
        description="Location of the homicide",
        default=None)
    county: County | None = Field(
        description="County where the homicide occurred (optional)",
        default=None)
    town: str | None = Field(
        description="Town or city where the homicide occurred (optional)",
        default=None)


class GenericArticleAnalysisResponse(BaseModel):
    """
    Generic response model for extracting information about victims
    """
    homicide_victims: List[GenericVictim] = Field(
        description="Information for each homicide victim in the article. "
        "Only include dead victims.")


class ArticleGenericVictimItem(NamedTuple):
    """
    Represents a potential victim extracted from an article
    """
    record_id: int
    victim: GenericVictim


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
    selected_homicides: tuple[int, ...] = tuple()
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
    # current_step: str = 'not_started'
    next_step: str = 'begin'
    victims: tuple[ArticleGenericVictimItem, ...] = tuple()


# An Action is a function that has State as argument,
#   performs a side effect, and returns an updated State
Action = Callable[[State], State]


# The response of a reaction (RxResp) is a tuple,
#   which includes the next Action to take, and the updated State
# RxResp = Tuple[Callable[...,State], State]
RxResp = tuple[Action, State]


# A reaction (no side effects) takes a State
#   and returns a RxResp (see above)
Reaction = Callable[[State], RxResp]
