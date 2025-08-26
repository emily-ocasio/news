"""
Defines functions with side effects and maps them to intents
"""
from dataclasses import dataclass
from typing import Callable
from .string import String

class Prompt(String):
    """
    Represents a prompt for user input.
    """
@dataclass(frozen=True)
class PutLine:
    """Base I/O: output a line."""
    s: str
    end: str = '\n'


@dataclass(frozen=True)
class GetLine:
    """Base I/O: input a line with prompt."""
    prompt: Prompt

REAL_DISPATCH: dict[type, Callable] = {}

def intentdef(intent: type) -> Callable[[Callable], Callable]:
    """
    Decorator for intent functions
    Registers the function in the REAL_DISPATCH dictionary
    """
    def decorator(func: Callable) -> Callable:
        REAL_DISPATCH[intent] = func
        return func
    return decorator

@intentdef(PutLine)
def _putline(x: PutLine) -> None:
    """
    Print a message to the console
    """
    print(x.s, end=x.end)

@intentdef(GetLine)
def _getline(x: GetLine) -> String:
    """
    Get a line of input from the user
    """
    return String(input(x.prompt + " > "))
