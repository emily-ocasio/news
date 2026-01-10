"""
Main menu constructor
"""

from enum import Enum
from pymonad import Run, Tuple, pure, put_line
from appstate import AppState
from choose import initial_prompts as mainmenu_prompts
from menuprompts import MenuChoice, MenuPrompts, input_from_menu, NextStep

from incidents import gpt_incidents
from incidents_dedupe import dedupe_incidents
from unnamed_match import match_unnamed_victims
from gpt_filtering import second_filter
from incidents_setup import build_incident_views
from fixarticle import fix_article
from geocode_incidents import geocode_incidents
from shr_match import match_article_to_shr_victims
from special_case_review import review_special_cases


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
    INCIDENTS = MenuChoice("I")
    GEOCODE = MenuChoice("M")
    DEDUP = MenuChoice("D")
    UNNAMED = MenuChoice("U")
    LINK = MenuChoice("L")
    SPECIAL = MenuChoice("P")
    QUIT = MenuChoice("Q")


class AfterTick(Tuple[AppState, NextStep]):
    """
    Represents the state and the next step after the tick.
    """

    @classmethod
    def make(cls, fst, snd) -> "AfterTick":
        return cls(fst, snd)

    @property
    def state(self) -> AppState:
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


def dispatch_from_main_menu(choice: MainChoice) -> Run[NextStep]:
    """
    Dispatch the 'tock' action based on the main result.
    """
    match choice:
        case MainChoice.GPT:
            return second_filter()
        case MainChoice.FIX:
            return fix_article()
        case MainChoice.EXTRACTION:
            return gpt_incidents()
        case MainChoice.INCIDENTS:
            return build_incident_views()
        case MainChoice.GEOCODE:  # <-- add
            return geocode_incidents()
        case MainChoice.DEDUP:
            return dedupe_incidents()
        case MainChoice.UNNAMED:
            return match_unnamed_victims()
        case MainChoice.LINK:
            return match_article_to_shr_victims()
        case MainChoice.SPECIAL:
            return review_special_cases()
        case (
            MainChoice.REVIEW
            | MainChoice.NEW
            | MainChoice.ASSIGN
            | MainChoice.AUTO
            | MainChoice.HUMANIZE
            | MainChoice.VICTIM
        ):
            return put_line(f"Dispatching to {choice.name}...") ^ pure(
                NextStep.CONTINUE
            )
        case MainChoice.QUIT:
            return pure(NextStep.QUIT)


def main_menu_tick() -> Run[NextStep]:
    """
    Display and select from main menu
    """
    return put_line("Main Menu:") ^ input_from_menu(MenuPrompts(mainmenu_prompts)) >> (
        lambda choice: dispatch_from_main_menu(MainChoice(choice))
    )
