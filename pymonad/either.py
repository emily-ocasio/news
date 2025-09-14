"""
Implementation of Either monad
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeVar, Callable, overload

from .functor import Functor, map # pylint:disable=redefined-builtin
from .monad import ap

L = TypeVar("L")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")

type Either[L,R] = 'Left[L]' | 'Right[R]'

@dataclass(frozen=True)
class Left[L](Functor):
    """
    Represents a left value in an Either type.
    """
    l: L

    @classmethod
    def make(cls, value) -> Left[L]:
        """Creates a new instance of Left."""
        return cls(value)

    def __mul__(self, other: Either[L, Any]) -> Left[L]:
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def map(self, f: Callable[[R], S]) -> Left[L]:
        return self.make(self.l)

    def _apply(self, other) -> "Left[L]":  # pylint: disable=unused-argument
        return self

    def __rshift__(self, m: Callable[[R], Either[L, S]]) -> Left[L]:
        return self

    def _bind(self, m: Callable[[R], Either[L, S]]) -> Left[L]:  # pylint: disable=unused-argument
        return self

    def __rand__(self, other: Callable[[R], S]) -> Left[L]:
        return map(other, self)

    def __repr__(self):
        """String representation of the Left."""
        return f"Left({self.l})"

    def __eq__(self, other) -> bool:
        """Equality check for Left."""
        return isinstance(other, Left) and self.l == other.l

@dataclass(frozen=True)
class Right[R](Functor[R]):
    """
    Represents a right value in an Either type.
    """
    r: R

    @overload
    def __mul__(self: Right[Callable[[S], T]], other: Left[L]) -> Left[L]: ...
    @overload
    def __mul__(self: Right[Callable[[S], T]], other: Right[S])\
        -> Right[T]: ...

    def __mul__(self: Right[Callable[[S], T]], other: Either[L, S])\
        -> Either[L, T]:
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    @classmethod
    def make(cls, value) -> Right:
        """Creates a new instance of Right."""
        return cls(value)

    def map(self, f: Callable[[R], S]) -> Right[S]:
        return self.make(f(self.r))

    def _apply(self: Right[Callable[[S], T]], other: Either[L, S])\
        -> Either[L, T]:
        """Applies the function wrapped in Right to another Either value."""
        return ap(self, other, Right)
        # match other:
        #     case Right(x):
        #         return self.make(self.r(x))
        #     case Left(l):
        #         return other

    def __rshift__(self, m: Callable[[R], Either[L, S]]) -> Either[L, S]:
        """
        Chains computations by passing the value inside Right to function m.
        """
        return self._bind(m)


    def _bind(self, m: Callable[[R], Either[L, S]]) -> Either[L, S]:
        return m(self.r)

    @classmethod
    def pure(cls, value: R) -> Either[Any, R]:
        """
        Wraps a value in the Right context.
        """
        _result: Either[Any, R] = Right.make(value)
        return _result

    def __rand__(self, other: Callable[[R], S]) -> Right[S]:
        return map(other, self)

    def __repr__(self):
        """String representation of the Right."""
        return f"Right({self.r})"

    def __eq__(self, other) -> bool:
        """Equality check for Right."""
        return isinstance(other, Right) and self.r == other.r

# el = Left("error")
# er = Right(42)
# f: Callable[[int], int] = lambda x: x + 1
# ef = Right(f)

# e1 = ef * er
# e2 = ef * el
# n =  Nothing
# #ex = ef * n     #error expected cannot apply Either to Nothing

# def apply(f, other):
#     return f * other

# e3 = apply(ef, er)
# e4 = apply(ef, el)
# e5 = apply(ef, n)
