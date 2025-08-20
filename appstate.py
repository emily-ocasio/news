"""
Defines monadic state at the top level.
"""
from dataclasses import dataclass

from pymonad import Lens, lens

@dataclass(frozen=True)
class AppState:
    """
    Application top-level state.
    """
    user_name: str = ""

user_name: Lens[AppState, str] = lens("user_name")
