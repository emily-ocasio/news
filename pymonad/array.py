""" Implements purescript-like Array type in Python."""
# pylint: disable=W0212
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Callable, TypeVar, Type, Self

from .applicative import Applicative
from .curry import curry2
from .functor import Functor
from .monoid import Monoid
from .monad import unit, Unit

B = TypeVar('B')
C = TypeVar('C')
M = TypeVar('M', bound=Monoid)
# F = TypeVar('F', bound=Applicative)

@dataclass(frozen=True)
class Array[A](Functor[A], Monoid):
    """
    Represents an immutable array.
    """
    a: tuple[A, ...]

    def __iter__(self):
        """Iterates over the elements of the Array."""
        return iter(self.a)

    def __add__(self: "Array[A]", other: "Array[A]") -> "Array[A]":
        """
        Overloads + operator to concatenate two Arrays.
        __iter__ and __add__ together allow for sum to work correctly.
        """
        return self.alt(other)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index: int) -> A:
        return self.a[index]

    @property
    def length(self) -> int:
        """Returns the length of the Array."""
        return len(self.a)

    def append(self: "Array[A]", other: "Array[A]") -> "Array[A]":
        return Array(self.a + other.a)

    def alt(self: "Array[A]", other: "Array[A]") -> "Array[A]":
        """
        Implements the alternative operation for Array.
        Returns a new Array containing elements from both Arrays.
        """
        return self.append(other)

    @classmethod
    def cons(cls, x: B, ar: "Array[B]") -> "Array[B]":
        """Prepends an element to the Array."""
        return Array((x,) + ar.a)

    @classmethod
    def snoc(cls, ar: Self, x: A) -> Self:
        """Appends an element to the Array."""
        return cls(ar.a + (x,))

    @classmethod
    def make(cls, a: tuple[A, ...]) -> 'Array[A]':
        """Creates a new instance of Array with the given elements."""
        return cls(a)

    @classmethod
    def mempty(cls) -> "Array":
        """Returns an empty Array."""
        return Array(())

    @classmethod
    def empty(cls) -> "Array":
        """Implementation of Plus typeclass for Array."""
        return Array(())

    def __rand__(self, other: Callable[[A], B]) -> 'Array[B]':
        """Defines the right-hand side of the map operation."""
        return self.map(other)

    def map(self, f: Callable[[A], B]) -> 'Array[B]':
        return Array(tuple(map(f, self.a)))

    def map_with_index(self, f: Callable[[int, A], B]) -> 'Array[B]':
        """
        Maps a function over the Array with access to each element's index.
        """
        return Array(tuple(f(i, x) for i, x in enumerate(self.a)))

    def __mul__(self: "Array[Callable[[B], C]]", other: "Array[B]") \
        -> 'Array[C]':
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def _concat(self: "Array[Array[B]]") -> 'Array[B]':
        """Concatenates an array of arrays into a single array."""
        return sum(self, Array.mempty())

    def _apply(self: "Array[Callable[[B], C]]", other: "Array[B]") \
        -> 'Array[C]':
        """
        Applies a function wrapped in the Array to a value wrapped in Array.
        Creates a new Array which is a cartesian product of the arrays
        """
        return ((lambda f: f & other) & self)._concat()

    @classmethod
    def pure(cls, x: A) -> 'Array[A]':
        """Creates a new Array instance with the one given element."""
        return cls((x,))

    def __rshift__(self, m: Callable[[A], "Array[B]"]) -> "Array[B]":
        """Defines the right-hand side of the bind operation."""
        return self._bind(m)

    def _bind(self, m: Callable[[A], "Array[B]"]) -> "Array[B]":
        """
        Chains computations by passing each element of the Array to function m.
        Returns a new Array containing the results.
        """
        return (m & self)._concat()

    def __repr__(self):
        """String representation of the Array."""
        return f"[{', '.join(map(repr, self.a))}]"

    def __contains__(self, item: A) -> bool:
        """
        Membership test: allows "item in my_array".
        """
        return item in self.a

    def foldl(self, f: Callable[[B, A], B], acc: B) -> B:
        """
        Left fold over the Array.
        Applies the function f to each element and an accumulator.
        """
        return reduce(f, self.a, acc)

    def foldr(self, f: Callable[[A, B], B], acc: B) -> B:
        """
        Right fold over the Array.
        Applies the function f to each element and an accumulator.
        """
        return reduce(lambda x, y: f(y, x), reversed(self.a), acc)

    def foldmap(self, f: Callable[[A], M], m_cls: Type[M]) -> M:
        """
        Maps each element of the Array to a Monoid and combines them.
        Returns the combined result.
        """
        return reduce(lambda x, y: x.append(f(y)), self.a, m_cls.mempty())

    def filter(self, predicate: Callable[[A], bool]) -> Array[A]:
        """
        Filters the Array based on a predicate function.
        Returns a new Array containing only elements that satisfy the predicate.
        """
        return Array(tuple(filter(predicate, self.a)))

    def traverse(self, f: Callable[[A], Applicative[B]]) \
        -> Applicative["Array[B]"]:
        """
        Traverses the Array, applying an applicative context 
        function to each element.
        Returns the combined result as an Array within the applicative context.
        """
        if not self.a:
            raise ValueError("Cannot traverse an empty Array.")

        # Start with the rightmost element
        # lifts an Applicative [B] into an Array of 1
        pure: "Callable[[B], Array[B]]" = \
            lambda b: self.__class__.cons(b, self.__class__.mempty())
        # Applicative[Array[B]] with only the last element
        acc0 = f(self.a[-1]).map(pure)

        # Array of all the elements except the last
        rest = self.make(self.a[:-1])

        def fn(x: A, acc: Applicative["Array[B]"]) -> Applicative["Array[B]"]:
            """
            Given the next element x, we want to apply f and 
            lift it into the applicative context
            """
            # curries function to add an element to the Array
            cons = curry2(self.__class__.cons)
            return (cons & f(x)) * acc
        return rest.foldr(fn, acc0)

    def traverse_(self, f: Callable[[A], Applicative[B]]) \
        -> Applicative[Unit]:
        """
        Traverses the Array, applying an applicative context function
        to each element and discarding results.
        """
        return self.traverse(f).map(lambda _: unit)

    def sequence(self: Array[Applicative[B]]) -> Applicative[Array[B]]:
        """
        Sequences an Array of Applicatives into an Applicative of Array.
        Applies the applicative context to each element and combines them.
        """
        return self.traverse(lambda x: x)

    def __eq__(self, other) -> bool:
        """Equality check for Array."""
        return isinstance(other, Array) and self.a == other.a

    @classmethod
    def replicate(cls, n: int, x: A) -> "Array[A]":
        """Creates an Array by replicating the given element n times."""
        return cls((x,) * n)
