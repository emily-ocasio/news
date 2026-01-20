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

def array_traverse_run(xs: Array[A], f: Callable[[A], Run[B]]) -> Run[Array[B]]:
    """
    Stack-safe traverse specialized for Run.
    Forces each Run step in a loop to avoid deep bind chains.
    """
    if not xs.a:
        raise ValueError("Cannot traverse an empty Array.")

    def step(self_run: Run[object]) -> Array[B]:
        acc = Array.mempty()
        for x in xs:
            b = f(x)._step(self_run) #pylint: disable=protected-access
            acc = Array.snoc(acc, b)
        return acc

    return Run(step, lambda intent, current: current._perform(intent, current)) #pylint: disable=protected-access

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
