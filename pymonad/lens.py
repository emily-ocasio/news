"""
Code to define lenses for accessing and modifying monadic state.
"""

from dataclasses import dataclass, replace
from typing import Callable, TypeVar

from .run import Run, get, put

S = TypeVar("S")  # whole state (a frozen dataclass)
T = TypeVar("T")  # focused field type

@dataclass(frozen=True)
class Lens[S, T]:
    """
    Allows for focused access and modification of a specific field
    within a larger state.
    """
    get: Callable[[S], T]  # function to get the field value
    set: Callable[[S, T], S]  # function to set the field value


# --- Lift lenses into Run API ---

def view(l: Lens[S, T]) -> "Run[T]":
    """
    Get a specific value from the state.
    """
    return l.get & get()

def set_(l: Lens[S, T], v: T) -> "Run[None]":
    """
    Set a specific value in the state.
    """
    return get() >> (lambda s: put(l.set(s, v)))

def over(l: Lens[S, T], f: Callable[[T], T]) -> "Run[None]":
    """
    Modify the state value using a function.
    """
    return get() >> (lambda s: put(l.set(s, f(l.get(s)))))

# Convenience: whole-state update
def modify(f: Callable[[S], S]) -> "Run[None]":
    """
    Modify state as a whole using a function
    """
    return get() >> (lambda s: put(f(s)))

def lens(field_name: str) -> Lens:
    """
    Create a lens for accessing a specific field in AppState.
    """
    return Lens(
        get=lambda s: getattr(s, field_name),
        set=lambda s, v: replace(s, **{field_name: v})
    )
