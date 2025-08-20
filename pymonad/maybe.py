""" Implementation of Maybe in Python."""
from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, EnumMeta
from dataclasses import dataclass
from typing import cast, Callable, TypeVar, overload

from .functor import Functor, map
from .monad import ap

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
F = TypeVar("F", bound='Maybe')


# class _Maybe[A](Apply['_Maybe', A]):
#     pass

type Maybe[A] = Just[A] | _Nothing


# class _Maybe[A](Functor[A], Apply[A], Applicative[A], Bind['Maybe', A], ABC):

#     def __rand__(self, other: Callable[[A], B]) -> Maybe[B]:
#         return map(other, self)

#     def __mul__(self: "_Maybe[Callable[[B],C]]", other: "_Maybe[B]") -> Maybe[C]:
#         return apply(self, other)
    
#     @abstractmethod
#     def map(self, f: Callable[[A], B]) -> Maybe[B]:
#         """Applies a function to the value inside Maybe."""

#     @abstractmethod
#     def apply(self: Apply[Callable[[B], C]], other: Apply[B]) \
#         -> Apply[C]:
#         """Applies a function wrapped in Maybe to a value wrapped in Maybe."""

#     @abstractmethod
#     def bind(self, m: Callable[[A], "Bind['Maybe', B]"]) -> "Maybe[B]":
#         """Chains computations by passing the value inside Maybe to function f."""

#     @classmethod
#     def pure(cls, value: A) -> Applicative[A]:
#         """Wraps a value in tyhhe Just context."""
#         return Just(value)

class NothingMaybeMeta(ABCMeta, EnumMeta):
    pass


class _Nothing(Functor, Enum, metaclass=NothingMaybeMeta):
    NOTHING = "Nothing"

    def __rand__(self, other: Callable[[A], B]) -> "_Nothing":
        return Nothing

    def map(self, f: Callable[[A], B]) -> "_Nothing":
        return Nothing

    def __mul__(self, other: Maybe) -> "_Nothing":
        return Nothing

    def _apply(self, other: Maybe) -> "_Nothing":
        return Nothing

    def __rshift__(self, m: Callable[[A], "Maybe[B]"]) -> "_Nothing":
        return Nothing

    def _bind(self, m: Callable[[A], "Maybe[B]"]) -> "_Nothing":
        return Nothing

    def __repr__(self):
        """String representation of Nothing."""
        return "Nothing"
    
    def pure(self, value):
        return Nothing

    def __eq__(self, other) -> bool:
        """Equality check for Nothing."""
        return isinstance(other, _Nothing)

# singleton instance
Nothing: _Nothing = _Nothing.NOTHING

@dataclass(frozen=True)
class Just[A](Functor[A]):
    a: A

    @classmethod
    def make(cls, value) -> 'Just':
        return Just(value)

    def map(self, f: Callable[[A], B]) -> "Just[B]":
        return self.make(f(self.a))

    def __rand__(self, other: Callable[[A], B]) -> "Just[B]":
        """Defines the right-hand side of the map operation."""
        return map(other, self)

    @overload
    def __mul__(self: "Just[Callable[[B], C]]", other: "Just[B]") -> "Just[C]": ...
    @overload
    def __mul__(self: "Just[Callable[[B], C]]", other: _Nothing) -> "_Nothing": ...
    @overload
    def __mul__(self: "Just[Callable[[B], C]]", other: "Maybe[B]") -> "Maybe[C]": ...


    def __mul__(self: "Just[Callable[[B], C]]", other: "Maybe[B]") -> Maybe[C]:
        return self._apply(other)

    def _apply(self: "Just[Callable[[B], C]]", other: Maybe[B]) -> Maybe[C]:
        """Applies a function wrapped in Just to a value wrapped in Maybe."""
        # match other:
        #     case _Nothing():
        #         return Nothing
        #     case Just(value):
        #         return self.make(self.a(value))
        return ap(self, other, Just)

    def __rshift__(self, m: Callable[[A], Maybe[B]]) -> Maybe[B]:
        """Chains computations by passing the value inside Just to function m."""
        return self._bind(m)

    def _bind(self, m: Callable[[A], Maybe[B]]) -> Maybe[B]:
        return m(self.a)

    def __repr__(self):
        """String representation of the Just."""
        return f"Just({self.a})"

    def __eq__(self, other) -> bool:
        """Equality check for Just."""
        return isinstance(other, Just) and self.a == other.a
    
    @classmethod
    def pure(cls, value: A) -> 'Just[A]':
        """Wraps a value in the Just context."""
        return cls(value)

# def test_maybe(a: Functor, b: Applicative) -> None:
#     """Test function for Maybe."""
#     print(a, b)

# a = Nothing
# b = Nothing

# test_maybe(a, b)
def fromMaybe(default: A, m: Maybe[A]) -> A:
    """Extracts the value from a Maybe, or returns a default value."""
    match m:
        case Just(value):
            return value
        case Nothing:
            return default