"""
Main menu constructor
"""

from enum import Enum
from pymonad import ErrorPayload, Run, Tuple, ask, pure, put_line, throw
from pymonad.run import StateRegistry
from appstate import AppState
from choose import initial_prompts as mainmenu_prompts
from menuprompts import MenuChoice, MenuPrompts, input_from_menu, NextStep

from incidents import gpt_incidents
from incidents_dedupe import dedupe_incidents
from splink_charts import splink_charts
from unnamed_match import match_unnamed_victims
from gpt_filtering import second_filter
from first_filter import first_filter
from incidents_setup import build_incident_views
from fixarticle import fix_article
from geocode_incidents import geocode_incidents
from shr_match import match_article_to_shr_victims
from special_case_review import review_special_cases
from special_add import add_special_articles
from vector_similarity import vector_similarity
from orphan_adjudication_apply import apply_orphan_adjudications
from orphan_adjudication_controller import adjudicate_orphans_controller
from orphan_postadj_cluster import cluster_postadj_orphans
from humanization_pipeline import humanization_shr
from review_humanization_determinations import review_humanization_determinations
from publication_profiles import (
    Availability,
    PublicationCapabilities,
    PublicationProfile,
)

MAIN_MENU_PROMPTS = mainmenu_prompts + (
    "Apply orphan adjudication [J]",
    "Adjudicate unmatched orphans [K]",
)


class MainChoice(Enum):
    """
    Main menu choices
    """

    REVIEW = MenuChoice("R")
    FIX = MenuChoice("F")
    NEW = MenuChoice("N")
    ASSIGN = MenuChoice("H")
    AUTO = MenuChoice("A")
    GPT = MenuChoice("S")
    EXTRACTION = MenuChoice("G")
    HUMANIZE = MenuChoice("Z")
    VICTIM = MenuChoice("V")
    VECTOR_SIM = MenuChoice("X")
    INCIDENTS = MenuChoice("I")
    GEOCODE = MenuChoice("M")
    DEDUP = MenuChoice("D")
    CHARTS = MenuChoice("C")
    UNNAMED = MenuChoice("U")
    POSTADJ_ORPHAN_CLUSTER = MenuChoice("O")
    ADJUDICATION_APPLY = MenuChoice("J")
    ADJUDICATION_CONTROLLER = MenuChoice("K")
    LINK = MenuChoice("L")
    SPECIAL = MenuChoice("P")
    SPECIAL_ADD = MenuChoice("Y")
    QUIT = MenuChoice("Q")


class AfterTick(Tuple[StateRegistry[AppState], NextStep]):
    """
    Represents the state and the next step after the tick.
    """

    @classmethod
    def make(cls, fst, snd) -> "AfterTick":
        return cls(fst, snd)

    @property
    def state(self) -> StateRegistry[AppState]:
        """
        State of the application
        """
        return self.fst

    @property
    def next_step(self) -> NextStep:
        """
        User's choice from the main menu
        """
        return self.snd


def _dispatch_available(choice: MainChoice) -> Run[NextStep]:
    """
    Dispatch the 'tock' action based on the main result.
    """
    match choice:
        case MainChoice.AUTO:
            action = first_filter()
        case MainChoice.GPT:
            action = second_filter()
        case MainChoice.FIX:
            action = fix_article()
        case MainChoice.EXTRACTION:
            action = gpt_incidents()
        case MainChoice.INCIDENTS:
            action = build_incident_views()
        case MainChoice.VECTOR_SIM:
            action = vector_similarity()
        case MainChoice.GEOCODE:  # <-- add
            action = geocode_incidents()
        case MainChoice.DEDUP:
            action = dedupe_incidents()
        case MainChoice.CHARTS:
            action = splink_charts()
        case MainChoice.UNNAMED:
            action = match_unnamed_victims()
        case MainChoice.POSTADJ_ORPHAN_CLUSTER:
            action = cluster_postadj_orphans()
        case MainChoice.ADJUDICATION_APPLY:
            action = apply_orphan_adjudications()
        case MainChoice.ADJUDICATION_CONTROLLER:
            action = adjudicate_orphans_controller()
        case MainChoice.LINK:
            action = match_article_to_shr_victims()
        case MainChoice.SPECIAL:
            action = review_special_cases()
        case MainChoice.SPECIAL_ADD:
            action = add_special_articles()
        case MainChoice.HUMANIZE:
            action = humanization_shr()
        case (
            MainChoice.NEW
            | MainChoice.ASSIGN
            | MainChoice.VICTIM
        ):
            action = put_line(f"Dispatching to {choice.name}...") ^ pure(
                NextStep.CONTINUE
            )
        case MainChoice.REVIEW:
            action = review_humanization_determinations()
        case MainChoice.QUIT:
            action = pure(NextStep.QUIT)
    return action


def _availability_for_choice(
    capabilities: PublicationCapabilities, choice: MainChoice
) -> Availability:
    """Read the capability availability required by a menu operation."""
    match choice:
        case MainChoice.AUTO:
            availability = capabilities.first_filter
        case MainChoice.GPT:
            availability = capabilities.gpt_classification
        case MainChoice.EXTRACTION:
            availability = capabilities.incident_extraction
        case MainChoice.INCIDENTS | MainChoice.VECTOR_SIM:
            availability = capabilities.incident_staging
        case MainChoice.GEOCODE:
            availability = capabilities.geocoding
        case MainChoice.DEDUP | MainChoice.CHARTS:
            availability = capabilities.named_victim_deduplication
        case MainChoice.UNNAMED:
            availability = capabilities.orphan_linkage
        case (
            MainChoice.POSTADJ_ORPHAN_CLUSTER
            | MainChoice.ADJUDICATION_APPLY
            | MainChoice.ADJUDICATION_CONTROLLER
        ):
            availability = capabilities.orphan_adjudication
        case MainChoice.LINK | MainChoice.HUMANIZE:
            availability = capabilities.shr_linkage
        case _:
            availability = capabilities.article_selection
    return availability


def _dispatch_for_profile(
    profile: PublicationProfile, choice: MainChoice
) -> Run[NextStep]:
    """Block unavailable publication operations before controller dispatch."""
    if choice is MainChoice.QUIT:
        return _dispatch_available(choice)
    availability = _availability_for_choice(profile.capabilities, choice)
    if availability is Availability.UNAVAILABLE:
        return throw(
            ErrorPayload(
                f"{profile.display_name}: selected operation is unavailable."
            )
        )
    return _dispatch_available(choice)


def dispatch_from_main_menu(choice: MainChoice) -> Run[NextStep]:
    """Dispatch a menu choice within the active publication profile."""
    return ask() >> (lambda env: _dispatch_for_profile(
        env["publication_profile"], choice
    ))


def main_menu_tick() -> Run[NextStep]:
    """
    Display and select from main menu
    """
    return ask() >> (lambda env:
        put_line(
            f"Main Menu — {env['publication_profile'].session_label}:"
        ) ^ input_from_menu(MenuPrompts(MAIN_MENU_PROMPTS)) >> (
            lambda choice: dispatch_from_main_menu(MainChoice(choice))
        )
    )
