"""
Helper functions and types related to menu prompts
"""
from collections.abc import Callable
from enum import Enum
from typing import Any

from calculations import unified_prompt
from pymonad import Array, InputPrompt, Char, Tuple, Run, get_line, pure

class NextStep(Enum):
    """
    Trampoline actions to take after the tick
    """
    CONTINUE = "continue"
    QUIT = "quit"

class MenuChoice(Char):
    """
    Represents a single choice in a menu
    """

class MenuDispatch(Tuple[MenuChoice, Callable[[Any], Run]]):
    """
    Represents a dispatch entry for a menu choice
    """

type MenuChoices = Array[MenuChoice]

class MenuPrompts(Array[InputPrompt]):
    """
    List of prompt options for a single menu
    """
    _full_prompt: InputPrompt
    _choices: MenuChoices
    _allow_return: bool
    _frozen: bool

    def __init__(self, prompts: tuple[str, ...],
                 add_quit: bool = True,
                 allow_return: bool = False,
                 width: int = 150):
        super().__init__(tuple(InputPrompt(p) for p in prompts))
        full_prompt, choices = unified_prompt(self.a, width=width,
            add_quit=add_quit, allow_return=allow_return)
        object.__setattr__(self, "_full_prompt", InputPrompt(full_prompt))
        object.__setattr__(self, "_choices",
                Array(tuple(MenuChoice(choice) for choice in choices)))
        object.__setattr__(self, "_allow_return", allow_return)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("MenuPrompts is immutable")
        object.__setattr__(self, name, value)

    @property
    def full_prompt(self) -> InputPrompt:
        """
        Get the full prompt text, formatted properly in columns
        """
        return self._full_prompt

    @property
    def choices(self) -> MenuChoices:
        """
        Get the list of menu choices
        """
        return self._choices

    @property
    def allow_return(self) -> bool:
        """
        Get the allow_return flag
        """
        return self._allow_return

def input_from_menu(prompts: MenuPrompts) -> Run[MenuChoice]:
    """
    Get user input from a menu prompt.
    - Print the formatted menu once.
    - Take the first character of the entered line; if it matches a MenuChoice,
      return that choice. Otherwise, repeat the "Select option > " prompt.
    """
    # Prompt object used for the "Select option" line
    repeat_prompt = InputPrompt("> ")

    def loop(show_full: bool) -> Run[MenuChoice]:
        # Show the full menu only on the first iteration
        return \
            get_line(prompts.full_prompt if show_full else repeat_prompt) \
                >> (lambda s:
            # s is a String; take its first character (or empty string)
            _process_input(str(s))
        )

    def _process_input(line: str) -> Run[MenuChoice]:
        if len(line) == 0:
            return pure(MenuChoice("\n")) \
                if prompts.allow_return else loop(False)
        if (choice := MenuChoice(line[0].upper())) in prompts.choices:
            return pure(choice)
        return loop(False)

    # Start by showing the full prompt once
    return loop(True)
