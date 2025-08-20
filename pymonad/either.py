from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar, Callable, overload

from .functor import Functor, map
from .monad import ap

L = TypeVar("L")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")

type Either[L,R] = Left[L] | Right[R]

# class _Either[R](Apply['_Either', R], Functor[R]):
#     pass
# @dataclass(frozen=True)
# class _Either[L,R](Applicative['Either', R], Apply[R], Functor[R], Bind['Either', R], ABC):

#     def __rand__(self, other: Callable[[R], S]) -> 'Either[L, S]':
#         return map(other, self)

#     def __mul__(self: "_Either[L, Callable[[S], T]]", other: "_Either[L, S]") -> 'Either[L, T]':
#         return apply(self, other)

#     @abstractmethod
#     def map(self, f: Callable[[R], S]) -> Either[L, S]:
#         ...

#     @abstractmethod
#     def apply(self: Apply[Callable[[S], T]], other: Apply[S]) -> Apply[T]: 
#         ...

#     @abstractmethod
#     def bind(self, m: Callable[[R], "Bind['Either', S]"]) -> Bind['Either', S]:
#         """Chains computations by passing the value inside Either to function m."""

#     @classmethod
#     def pure(cls, t: Type, value: R) -> Applicative['Either', R]:
#         """Wraps a value in the Right context."""
#         return Right.make(value)


@dataclass(frozen=True)
class Left[L](Functor):
    l: L

    @classmethod
    def make(cls, value) -> 'Left':
        """Creates a new instance of Left."""
        return cls(value)

    def __mul__(self, other: Either[L, Any]) -> "Left[L]":
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def map(self, f: Callable[[R], S]) -> 'Left[L]':
        return self.make(self.l)

    def _apply(self, other) -> "Left[L]":
        return self
    
    def __rshift__(self, m: Callable[[R], Either[L, S]]) -> "Left[L]":
        return self

    def _bind(self, m: Callable[[R], Either[L, S]]) -> "Left[L]":
        return self

    def __rand__(self, other: Callable[[R], S]) -> 'Left[L]':
        return map(other, self)

    def __repr__(self):
        """String representation of the Left."""
        return f"Left({self.l})"
    
    def __eq__(self, other) -> bool:
        """Equality check for Left."""
        return isinstance(other, Left) and self.l == other.l

@dataclass(frozen=True)
class Right[R](Functor[R]):
    r: R

    @overload
    def __mul__(self: "Right[Callable[[S], T]]", other: Left[L]) -> Left[L]: ...
    @overload
    def __mul__(self: "Right[Callable[[S], T]]", other: "Right[S]") -> "Right[T]": ...

    def __mul__(self: "Right[Callable[[S], T]]", other: Either[L, S]) -> Either[L, T]:
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    @classmethod
    def make(cls, value) -> 'Right':
        """Creates a new instance of Right."""
        return cls(value)

    def map(self, f: Callable[[R], S]) -> 'Right[S]':
        return self.make(f(self.r))

    def _apply(self: "Right[Callable[[S], T]]", other: Either[L, S]) -> Either[L, T]:
        """Applies the function wrapped in Right to another Either value."""
        return ap(self, other, Right)
        # match other:
        #     case Right(x):
        #         return self.make(self.r(x))
        #     case Left(l):
        #         return other

    def __rshift__(self, m: Callable[[R], Either[L, S]]) -> Either[L, S]:
        """Chains computations by passing the value inside Right to function m."""
        return self._bind(m)


    def _bind(self, m: Callable[[R], Either[L, S]]) -> Either[L, S]:
        return m(self.r)

    @classmethod
    def pure(cls, value: R) -> 'Right[R]':
        return cls(value)

    def __rand__(self, other: Callable[[R], S]) -> 'Right[S]':
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