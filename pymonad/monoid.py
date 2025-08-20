# --------------------------------------------------------
# (c) Copyright 2014, 2020 by Jason DeLaat.
# Licensed under BSD 3-clause licence.
# --------------------------------------------------------
# pylint:disable=W2301
"""Monoid Implementation.

A monoid is an algebraic structure consisting of a set of objects, S,
and an operation usually denoted as '+' which obeys the following
rules:

    1. Closure: If 'a' and 'b' are in S, then 'a + b' is also in S.
    2. Identity: There exists an element in S (denoted 0) such that
       a + 0 = a = 0 + a
    3. Associativity: (a + b) + c = a + (b + c)

The monoid module provides a generic zero/identity element called
IDENTITY.

Monoid addition with IDENTITY simply always returns the other element
regardless of type.

Example:
    IDENTITY == IDENTITY # True.
    IDENTITY + 10      # 10
    'hello' + IDENTITY # 'hello'

"""

from typing import (
    Iterable,
    Protocol,
    Self,
)

from .semigroup import Semigroup

class Monoid(Semigroup, Protocol):
    """Base class for Monoid instances.

    To implement a monoid instance, create a sub-class of Monoid and
    override the identity_element and addition_operation methods
    ensuring that the closure, identity, and associativity laws hold.

    """

    # @classmethod
    # def wrap(cls, value) -> Self:
    #     """Wraps a value in the Monoid type."""
    #     if value is None:
    #         return cls.mempty()
    #     return cls(value)

    @classmethod
    def mempty(cls) -> Self:
        """Returns the identity element for this Monoid."""
        ...

# # class _MonoidIdentity[a : Monoid](a):
#  #once Python Types has those features, this is what we want
# class _MonoidIdentity[T](Monoid[T]):
#     superclass = Monoid

#     def __init__(self):
#         found = False
#         for i in type(self).__mro__:
#             if (
#                 i != _MonoidIdentity
#                 and i != self.__class__
#                 and i != Monoid
#                 and i != Generic
#                 and i != object
#             ):
#                 self.superclass = i
#                 found = True
#                 break
#         if not found and self.__class__ != _MonoidIdentity:
#             raise Exception("no superclass found")
#         self.value = None

#     def __add__(self: Self, other: Monoid[T] | T):
#         if not isinstance(other, Monoid):
#             return self.superclass(other)
#         return other

#     def __radd__(self, other: Self):
#         if not isinstance(other, Monoid):
#             return self.superclass(other)
#         return other

#     def __repr__(self):
#         return "IDENTITY"


# IDENTITY = _MonoidIdentity()


# mconcat([Monoida, Monoidb]) not throws an error :nice
def mconcat[M: Monoid](monoid_list: Iterable[M]) -> M:
    """Takes a list of monoid values and reduces them to a single value
    by applying the append operation to all elements of the list.
    Needs a non empty list, because Python doesn't allow calling on types
    """
    it = iter(monoid_list)

    # a.identity()
    result = next(it)
    for value in it:
        result = result.append(value)
    return result
