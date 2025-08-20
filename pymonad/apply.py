"""This module provides the Apply abstract type class."""
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, overload

from .maybe import Just, Nothing
from .either import Left, Right
from .semigroup import Semigroup
from .string import String
from .tuple import Tuple
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
L = TypeVar('L')
R = TypeVar('R')
S = TypeVar('S', bound=Semigroup)

# class ApplyL(Functor[A], Protocol[S, A]):
#     """
#     Base class for Apply instances with further constraint that the functor is a Semigroup.
#     """

#     def apply(self: "ApplyL[S, Callable[[A], B]]", other: "ApplyL[S,A]") -> "ApplyL[S, B]":
#         """Applies a function wrapped in Functor to a value inside the Apply."""
#         ...

# class Apply[F, A](ABC):
#     """
#     Base class for Apply instances.
#     """

#     # def __mul__(self: "Apply[Callable[[B], C]]", other: "Apply[B]") -> "Apply[C]":
#     #     """Overrides the * operator for use as apply."""
#     #     return apply(self, other)


#     @abstractmethod
#     def apply(self: "Apply[F, Callable[[B], C]]", other: "Apply[F, B]") -> "Apply[F, C]":
#         """Applies a value in context to a function in context."""

# @overload
# def apply(f: _Nothing, a: Maybe[A]) -> _Nothing: ...
# @overload
# def apply(f: Just[Callable[[A], B]], a: Just[A]) -> Just[B]: ...
# @overload
# def apply(f: Just[Callable[[A], B]], a: _Nothing) -> _Nothing: ...
# @overload
# def apply(f: Just[Callable[[A], B]], a: Maybe[A]) -> Maybe[B]: ... 
# @overload
# def apply(f: Left[L], a: Either[L, A]) -> Left[L]: ...
# @overload
# def apply(f: Right[Callable[[A], B]], a: Left[L]) -> Left[L]: ...
# @overload
# def apply(f: Right[Callable[[A], B]], a: Right[A]) -> Right[B]: ...
# @overload
# def apply(f: ApplyTuple[S, Callable[[A], B]], a: ApplyTuple[S, A]) -> ApplyTuple[S, B]: ...
# @overload
# def apply(f: Apply[F, Callable[[A], B]], a: Apply[F, A]) -> Apply[F, B]: ...


def apply(f, other):
    return f * other

j1 = Just(2)
f: Callable[[int], int] = lambda x: x + 1

j2 = Just(f)
n = Nothing
j3 = j2 * j1
jn = j2 * n
e2 = Right(f)
e1 = Right(2)

e3 = e2 * e1
el = Left("error")
e4 = e2 * el
e5 = el * e1
e6 = el * e2
e7 = apply(e2, e1)
e8 = apply(el, e1)

t1 = Tuple(String("asd"), 2)
t2 = Tuple(String("qwe"), f)

t3 = t2 * t1
t4 = apply(t2, t1)

t5 = f & t1
t6 = Tuple(String("xyz"), 4)


at1 = Tuple(String("abc"), 2)
at2 = Tuple(String("xyz"), f)
at3 = at2 * at1



#ex = e2 * n  # error expected cannot apply Either to Nothing
#jx = j2 * el  # error expected cannot apply Maybe to Left
#tx = t2 * n # error expected cannot apply ApplyTuple to Nothing