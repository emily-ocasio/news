""" monad protocol
"""
# pylint: disable=W2301
from typing import Any, Callable, Protocol, TypeVar, cast, \
    ParamSpec, Concatenate

from .applicative import Applicative
#from .maybe import Maybe

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

class Monad[A](Applicative[A], Protocol):
    """
    Protocol for Monad, extending Applicative
    with bind and right-shift operations.
    """
    def __rshift__(self, m: Callable):
        """ Override >>> operator """
        ...

    def _bind(self, m: Callable):
        """
        Chains computations by passing the value inside the Monad to function f.
        """
        ...
# @overload
# def composeKleisli(g: Callable[[B], Maybe[C]], f: Callable[[A], Maybe[B]])
#   -> Callable[[A], Maybe[C]]:...
# @overload
# def composeKleisli(g: Callable[[B], Monad[C]], f: Callable[[A], Monad[B]])
#   -> Callable[[A], Monad[C]]:...
def compose_kleisli(g: Callable[[B], Any], f: Callable[[A], Any]) \
    -> Callable[[A], Any]:
    """
    Composes two Kleisli functions.
    """
    return lambda x: f(x)._bind(g) # pylint: disable=W0212

class Kleisli:
    """ Kleisli functions """
    def __init__(self, func: Callable[[A], Any]):
        self.func: Callable[[Any], Monad] = cast(Callable[[Any], Monad], func)

    def __lshift__(self, other):
        # Compose self.func after other.func
        return Kleisli(compose_kleisli(self.func, other.func))

    def __call__(self, x):
        return self.func(x)
# @overload
# def ap(mf: Maybe, mx: Maybe, mtype: Just) -> Maybe:...
# @overload
# def ap(mf:Monad, mx: Monad, mtype: Monad) -> Monad:...
def ap(mf, mx, mtype):
    """ 
    Used to implement apply in terms of bind
    The binds provide the monadic logic that would need to be replicated 
    in both bind and apply
    """
    return \
        mf >> (lambda f:
        mx >> (lambda x:
        mtype.pure(f(x))
        ))

def wal(*args):
    """
    Helper function to leverage walrus operator inside bind function definitions
    Takes a number of expressions and returns only the last one
    This allows all the other expressions to define values
    using the := walrus operator
    """
    return args[-1]

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

def bind_first(func: Callable[Concatenate[T, P], R], first: T) \
    -> Callable[P, R]:
    """
    Bind the first positional argument of `func` and return a callable typed
    with the remaining parameters.
    This preserves parameter types for type checkers.
    """
    def _bound(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(first, *args, **kwargs)
    return _bound

def comp(f: Callable, g: Callable) -> Callable:
    """
    Composes two functions f and g into a single function.
    """
    return lambda x: f(g(x))

def const(x, _):
    """
    Returns a function that ignores its argument and always returns x.
    """
    return x

def identity(x):
    """
    Returns the argument unchanged.
    """
    return x
