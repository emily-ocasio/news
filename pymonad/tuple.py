"""
This module defines several Tuple concrete and abstract types. 
"""
from collections.abc import Callable
from typing import Iterator, Self, TypeVar, Type
from dataclasses import dataclass


from .monoid import Monoid
from .semigroup import Semigroup
from .functor import Functor, map # pylint:disable=W0622
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
F = TypeVar('F', bound="Tuple")
L = TypeVar('L')
S = TypeVar('S', bound=Semigroup)
T = TypeVar('T', bound=Semigroup)
M = TypeVar('M', bound=Monoid)
N = TypeVar('N', bound=Monoid)

# class Applicative[A]():
#     """ Base class for Applicative instances."""


@dataclass(frozen=True)
class Tuple[L,A](Functor[A]):
    """An immutable Tuple functor that wraps two values."""

    fst: L
    snd: A

    @classmethod
    def make(cls, fst, snd) -> 'Tuple':
        """Creates a new instance of the specific subtype."""
        return cls(fst, snd)

    def map(self: Self, f: Callable[[A], B]) -> "Tuple[L, B]":
        """Applies a function to the second value inside the Tuple."""
        return self.make(self.fst, f(self.snd))

    def __rand__(self, other: Callable[[A], B]):
        """Defines the right-hand side of the map operation."""
        return map(other, self)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[L | A]:
        yield self.fst
        yield self.snd

    def __getitem__(self, index: int):
        """Allows indexing into the Tuple."""
        match index:
            case 0: return self.fst
            case 1: return self.snd
            case _: raise IndexError("Tuple index out of range")

    def __mul__(self: "Tuple[S, Callable[[B], C]]", other: "Tuple[S, B]") \
        -> "Tuple[S, C]":
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def _apply(self: "Tuple[S, Callable[[B], C]]", other: "Tuple[S, B]") \
        -> "Tuple[S, C]":
        """
        Applies the function wrapped in the Tuple to the second value.
        The Semigroup instance is used to combine the first values.
        """
        return self.__class__.make(self.fst.append(other.fst),
                                   self.snd(other.snd))

    def __rshift__(self: "Tuple[S, A]", m: Callable[[A], "Tuple[S, B]"]) \
        -> "Tuple[S, B]":
        """Defines the right-hand side of the bind operation."""
        return self._bind(m)

    def _bind(self: "Tuple[S, A]", m: Callable[[A], "Tuple[S, B]"]) \
        -> "Tuple[S, B]":
        """
        Chains computations by passing the value inside Tuple to function m.
        The first value is unchanged.
        """
        match m(self.snd):
            case Tuple(fst, snd):
                return self.make(self.fst.append(fst), snd)
            case _:
                raise TypeError("Function must return a Tuple instance.")

    @classmethod
    def pure(cls, t: Type[Monoid], value: A) -> "Tuple[L, A]":
        """Wraps a value in the Tuple context."""
        return cls.make(t.mempty(), value)

    def __repr__(self):
        """String representation of the Tuple."""
        return f'Tuple ({self.fst}, {self.snd})'

    def __eq__(self, other) -> bool:
        """Equality check for Tuple."""
        return isinstance(other, Tuple) \
            and self.fst == other.fst and self.snd == other.snd

@dataclass(frozen=True)
class Threeple[L, A, B](Functor[A]):
    """An immutable Threeple functor that wraps three values."""

    fst: L
    snd: A
    trd: B

    @classmethod
    def make(cls, fst, snd, trd) -> 'Threeple':
        """Creates a new instance of the specific subtype."""
        return cls(fst, snd, trd)

    def __rand__(self, other: Callable[[A], C]) -> "Threeple[L, C, B]":
        """Defines the right-hand side of the map operation."""
        return map(other, self)

    def map(self: Self, f: Callable[[A], C]) -> "Threeple[L, C, B]":
        """Applies a function to the second value inside the Threeple."""
        return self.make(self.fst, f(self.snd), self.trd)

    def __mul__(self: "Threeple[S, Callable[[C], D], B]",
                other: "Threeple[S, C, B]") -> "Threeple[S, D, B]":
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def _apply(self: "Threeple[S, Callable[[C], D], B]",
               other: "Threeple[S, C, B]") -> "Threeple[S, D, B]":
        """ Applies the function wrapped in the Threeple to the second value.
        The Semigroup instance is used to combine the first values.
        """
        return self.__class__.make(self.fst.append(other.fst),
                                   self.snd(other.snd),
                                   self.trd)

    def __repr__(self):
        """String representation of the Threeple."""
        return f'Threeple ({self.fst}, {self.snd}, {self.trd})'

    def __eq__(self, other) -> bool:
        """Equality check for Threeple."""
        return isinstance(other, Threeple) \
            and self.fst == other.fst \
            and self.snd == other.snd \
            and self.trd == other.trd
