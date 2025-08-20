"""
Applicative functor protocol definitions for pymonad.
"""

# applicative.py
# pylint:disable=W2301
from typing import Callable, Protocol, TypeVar, runtime_checkable

F = TypeVar("F")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

@runtime_checkable
class Applicative(Protocol[B]):
    """
    Protocol for Applicative functors, providing methods for pure, map,
    and applicative application.
    """
    @classmethod
    def pure(cls, value: B) -> "Applicative[B]":
        """
        Wraps a value in the Applicative context.
        """
        ...

    def map(self, f: Callable[[B], C]) -> "Applicative[C]":
        """
        Applies a function to the value inside the Applicative context.
        """
        ...

    def _apply(self, other: "Applicative") -> "Applicative":
        """
        Applies the function wrapped in this Applicative context to the value
        in another Applicative context.
        """
        ...

    def __rand__(self, f: Callable[[B], C]) -> "Applicative[C]":
        """
        Enables using the & operator for mapping a function
        over the Applicative.
        """
        ...

    def __mul__(self, other: "Applicative") -> "Applicative":
        """
        Enables using the * operator for applicative application.
        """
        ...
