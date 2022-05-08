from typing import Callable, Tuple
from state import State
from functools import reduce
import actions

# An Action is a function that has State as argument,
#   performs a side effect, and returns an updated State
Action = Callable[[State], State]

# The response of a reaction (RxResp) is a tuple,
#   which includes the next Action to take, and the updated State
#RxResp = Tuple[Callable[...,State], State]
RxResp = Tuple[Action, State]

# A reaction (no side effects) takes a State
#   and returns a RxResp (see above)
Reaction = Callable[[State], RxResp]

def combine_actions(*acts: Action) -> Action:
    """
    Special function that combines multiple actions into one
    Each of the actions must be a function of the form:
        new_state = funct(current_state)
    The overall function takes a variable number of actions (*acts)
        as arguments.
    Then an inner function combined_action is defined with the correct
        signature as a closure (the actions of each of the acts are closed into
        the fuction). The overall glue that puts it together is reduce, which
        takes all of the acts, into a final function, then that function has to
        be called with state as the paramenter.
    Reduce, in turn, depends on the inner combine2, which take two functions
        and returns the composite of them
    """
    def combined_action(state: State) -> State:
        def combine2(f1, f2):
            return lambda x:f2(f1(x))
        return reduce(combine2, acts)(state)       
    return combined_action

def action2(name: str, *args, **kwargs) -> Action:
    """
    Helper function used by reactions
    Takes the name of an action, and positional and named parameters
    Returns a new Action that first inserts the arguments in the state
        and then runs the rest of the named action inside of the actions module
    """
    def return_action(state: State) -> State:
        state = state._replace(inputargs = args, inputkwargs = kwargs)
        return getattr(actions, name)(state)
    return return_action
