"""
Function overloads for Applicative traverse
"""
from typing import Callable, overload, TypeVar, Any
from .array import Array
from .applicative import Applicative
from .run import Run
from .validation import V

A = TypeVar("A")
B = TypeVar("B")

@overload
def array_traverse(xs: Array[A], f: Callable[[A], Run[B]]) -> Run[Array[B]]: ...
@overload
def array_traverse(xs: Array[A], f: Callable[[A], Applicative[B]]) \
    -> Applicative[Array[B]]: ...
def array_traverse(xs: Array[A], f: Callable[[A], Applicative[B]]) \
    -> Applicative[Array[B]]:
    """ 
    Single implementation delegates to Array.traverse
    overloads refine types
    """
    return xs.traverse(f)

@overload
def array_sequence(xs: Array[V[A, B]]) -> V[A, Array[B]]: ...
@overload
def array_sequence(xs: Array[Run[B]]) -> Run[Array[B]]: ...
@overload
def array_sequence(xs: Array[Applicative[B]]) -> Applicative[Array[B]]: ...
def array_sequence(xs: Array[Any]) -> Applicative[Array[Any]]:
    """
    Single implementation delegates to Array.sequence
    overloads refine types
    """
    return xs.sequence()
