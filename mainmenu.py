"""
Main menu constructor
"""
from enum import Enum
from pymonad import Run, Tuple, pure, put_line
from appstate import AppState
from choose import initial_prompts as mainmenu_prompts
from menuprompts import MenuChoice, MenuPrompts, input_from_menu

class MainChoice(Enum):
    """
    Main menu choices
    """
    REVIEW = MenuChoice('R')
    FIX = MenuChoice('F')
    NEW = MenuChoice('N')
    ASSIGN = MenuChoice('H')
    AUTO = MenuChoice('A')
    GPT = MenuChoice('S')
    HUMANIZE = MenuChoice('Z')
    VICTIM = MenuChoice('V')
    QUIT = MenuChoice('Q')

# class MainResult(Tuple[AppState, MainChoice]):
#     """
#     Main menu result
#     """
#     @classmethod
#     def make(cls, fst, snd) -> 'MainResult':
#         return cls(fst, snd)

#     @property
#     def state(self) -> AppState:
#         """
#         State of the application
#         """
#         return self.fst

#     @property
#     def choice(self) -> MainChoice:
#         """
#         User's choice from the main menu
#         """
#         return self.snd

class NextStep(Enum):
    """
    Trampoline actions to take after the tick
    """
    CONTINUE = "continue"
    QUIT = "quit"

class AfterTick(Tuple[AppState, NextStep]):
    """
    Represents the state and the next step after the tick.
    """
    @classmethod
    def make(cls, fst, snd) -> 'AfterTick':
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

def dispatch_from_main_menu(choice: MainChoice) \
    -> Run[NextStep]:
    """
    Dispatch the 'tock' action based on the main result.
    """
    match choice:
        case MainChoice.REVIEW \
            | MainChoice.FIX \
            | MainChoice.NEW \
            | MainChoice.ASSIGN \
            | MainChoice.AUTO \
            | MainChoice.GPT \
            | MainChoice.HUMANIZE \
            | MainChoice.VICTIM:
            return \
                put_line(f"Dispatching to {choice.name}...") ^ \
                pure(NextStep.CONTINUE)
        case MainChoice.QUIT:
            return pure(NextStep.QUIT)

def main_menu_tick() -> Run[NextStep]:
    """
    Display and select from main menu
    """
    return \
        put_line("Main Menu:") ^ \
        input_from_menu(MenuPrompts(mainmenu_prompts)) >> (lambda choice:
        dispatch_from_main_menu(MainChoice(choice))
        )
