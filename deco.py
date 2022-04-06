from pickle import TUPLE1
from typing import NamedTuple, Tuple, Optional
class State(NamedTuple):
    inputargs: Tuple = ()
    inputkwargs: dict = {}
    outputs: Optional[Tuple] = None

def actiondef(a):
    def wrapper(state: State):
        print("Wrapped")
        state = state._replace(outputs = a(*state.inputargs, **state.inputkwargs))
        return state
    return wrapper

@actiondef
def act1(arg1, arg2, x, y):
    print(f"Function arguments: {arg1}, {arg2}")
    return 3, x, y

@actiondef
def act2(arg1):
    print(f"One argument: {arg1}")
    return arg1+200

