"""
Defines monadic state at the top level.
"""
from dataclasses import dataclass, replace

from pymonad import Lens

@dataclass(frozen=True)
class AppState:
    """
    Application top-level state.
    """
    user_name: str = ""

user_name_lens = Lens[AppState, str](
    get=lambda s: s.user_name,
    set=lambda s, v: replace(s, user_name=v)
)
