"""
Defines monadic state at the top level.
"""
from dataclasses import dataclass, fields
from typing import Any, Self

from pymonad import Lens, lens, Monoid, String

@dataclass(frozen=True)
class AppState(Monoid):
    """
    Application top-level state.
    """
    user_name: String = String.mempty()
    selected_option: String = String.mempty()
    prompt_key: String = String.mempty()
    latest_splink_linker: Any | None = None

    @classmethod
    def mempty(cls) -> "AppState":
        """
        Create an empty AppState.
        """
        return cls()

    def append(self, other: Self) -> Self:
        """
        Last-nonempty overriding merge
        """
        def is_empty(x) -> bool:
            if isinstance(x, Monoid):
                return x == x.__class__.mempty
            return x in ("", None)

        out = {}
        for f in fields(self):
            self_v = getattr(self, f.name)
            other_v = getattr(other, f.name)
            out[f.name] = self_v if is_empty(other_v) else other_v
        return type(self)(**out)

user_name: Lens[AppState, str] = lens("user_name")
selected_option: Lens[AppState, str] = lens("selected_option")
prompt_key: Lens[AppState, str] = lens("prompt_key")
latest_splink_linker: Lens[AppState, Any | None] = lens("latest_splink_linker")
