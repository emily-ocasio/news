from functools import wraps
from typing import Callable, Type, TypeVar, get_type_hints

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
def curry2(f: Callable[[X, Y], Z]) -> Callable[[X], Callable[[Y], Z]]:
    """Curry a binary function into two unary functions."""
    return lambda a: lambda b: f(a, b)

def return_type(func: Callable) -> Type:
    """Get the return type of a function."""
    hints = get_type_hints(func)
    _type = hints.get('return')
    assert( _type is not None), "Function must have a return type hint"
    return _type

def curry3(f):
    """Curry a ternary function into three unary functions."""
    return lambda a: lambda b: lambda c: f(a, b, c)

def curryN(f):
    """Curry any function f of N arguments into a chain of unary functions."""
    @wraps(f)
    def curried(*args):
        if len(args) >= f.__code__.co_argcount:
            return f(*args)
        return lambda x: curried(*args, x)
    return curried