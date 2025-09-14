"""
Applicative functor protocol definitions for pymonad.
"""
from __future__ import annotations
# pylint:disable=W2301
from typing import Callable, Protocol, TypeVar, runtime_checkable, Self

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

@runtime_checkable
class Applicative(Protocol[T]):
    """
    Protocol for Applicative functors, providing methods for pure, map,
    and applicative application.
    """
    @classmethod
    def pure(cls, value: T) -> Self:
        """
        Wraps a value in the Applicative context.
        """
        ...

    def map(self, f: Callable[[T], U]) -> Applicative[U]:
    # def map(self, f: Callable) -> Self:
        """
        Applies a function to the value inside the Applicative context.
        """
        ...

    def _apply(self: Applicative[Callable[[U], V]], other: Applicative[U]) \
        -> Applicative[V]:
        """
        Applies the function wrapped in this Applicative context to the value
        in another Applicative context.
        """
        ...

    def __rand__(self, f: Callable[[T], U]) -> Applicative[U]:
    # def __rand__(self, f: Callable) -> Self:
        """
        Enables using the & operator for mapping a function
        over the Applicative.
        """
        ...

    def __mul__(self: Applicative[Callable[[U], V]], other: Applicative[U]) \
        -> Applicative[V]:
        """
        Enables using the * operator for applicative application.
        """
        ...
