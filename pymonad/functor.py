""" Abstract base class for Functor """
from abc import ABC, abstractmethod
from typing import Callable, Self, TypeVar

# pylint:disable=C0105
A = TypeVar('A', covariant=True)
B = TypeVar('B', covariant=True)
F = TypeVar('F')
G = TypeVar('G')

class Functor[A](ABC):
    """Base class for Functor instances.

    To implement a functor instance, create a sub-class of Functor and
    override the map method ensuring that the functor laws hold.
    """

    @abstractmethod
    def __rand__(self, other):
        """Defines the right-hand side of the map operation."""
        return map(other, self)


    @abstractmethod
    def map(self: Self, f: Callable[[A], B]) -> "Functor[B]":
        """Applies a function to the value inside the Functor."""

def map(fn, f):  # pylint:disable=W0622
    """Applies the function 'fn' to the value inside the functor
    'f' using its map method."""
    return f.map(fn)

# a = Just(2)
# ae = Left("ad")
# ae2 = Right(3)
# f: Callable[[int], int] = lambda x: x + 1
# g: Callable[[str], str] = lambda x: x.upper()
# b = map(f, a)
# be = map(f, ae)
# be2 = map(f, ae2)
# t = Tuple("asd", 2)
# ft = map(f, t)
# #at = ApplyTuple("asd", 2)
# #fat = map(f, at)
# n = Nothing

# am: Maybe

# #fat = f & at
# b = f & a
# bn = f & n
