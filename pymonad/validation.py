"""
Implements purescript-like Validation applicative in Python
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from .applicative import Applicative
from .either import Either, Left, Right
from .functor import Functor, map # pylint:disable=W0622
from .semigroup import Semigroup

S = TypeVar('S')
T = TypeVar('T')
E = TypeVar('E', bound=Semigroup)

Valid = Right
Invalid = Left
type Validity[E, R] = Invalid[E] | Valid[R]

@dataclass(frozen=True)
class V[E, R](Functor[R], Applicative[R]):
    """
    Applicative validation type that accumulates errors.
    """
    either: Either[E, R]

    @property
    def validity(self) -> Validity[E, R]:
        """ Access underlying Either value """
        return self.either

    def map(self, f: Callable[[R], S]) -> V[E, S]:
        """ Functor map delegated to Either """
        return V(self.either.map(f))

    def __mul__(self: V[E, Callable[[S], T]],
                other: Applicative[S]) -> V[E, T]:
        """Overrides the * operator for use as apply."""
        return self._apply(other)

    def __rand__(self, other: Callable[[R], S]) -> V[E, S]:
        """Defines the right-hand side of the map operation."""
        return map(other, self)

    def _apply(self: V[E, Callable[[S], T]],
               other: Applicative[S]) -> V[E, T]:
        """ Apply accumulates errors in Left, applies function in Right """
        other_v: V[E, S] = cast(V[E, S], other)
        # mypy has trouble narrowing types in pattern matching of tuples
        match self.validity:
            case Valid(f):
                match other_v.validity:
                    case Valid(x):
                        return V(Valid.make(f(x)))
                    case Invalid(err):
                        return V(Invalid.make(err))
            case Invalid(err1):
                match other_v.validity:
                    case Valid(_):
                        return V(Invalid.make(err1))
                    case Invalid(err2):
                        erra = cast(Semigroup, err1)
                        errb = cast(Semigroup, err2)
                        return V(Invalid.make(erra.append(errb)))

    @classmethod
    def pure(cls, value:T) -> V[E, T]:
        """ Wraps a value in a successful Validation """
        _result: V[E, T] = V(Valid.pure(value))
        return _result

    @classmethod
    def invalid(cls, error: E) -> V[E, R]:
        """ Wraps an error in a failed Validation """
        _result: V[E, R] = V(Invalid.make(error))
        return _result

    def is_valid(self) -> bool:
        """ Returns True if the Validation is valid (i.e., contains a Right) """
        return isinstance(self.either, Valid)

    def apply_second(self, other: V[E, T]) -> V[E, T]:
        """
        Sequences two Validations, discarding the value of the first.
        Accumulates errors if either is invalid.
        """
        def f(_: R) -> Callable[[T], T]:
            # Replaces the first result with an identity function
            return lambda y: y
        return (f & self) * other

    def __xor__(self, other: V[E, T]) -> V[E, T]:
        """
        Overrides the ^ operator to use apply_second.
        """
        return self.apply_second(other)
