"""
Utility functions used to refer to actions and reactions
"""
from functools import reduce, wraps
from typing import Callable
from state import State, Action, Reaction, RxResp
import actions


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
        def combine2(act1: Action, act2: Action) -> Action:
            return lambda x: act2(act1(x))
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
        state = state._replace(inputargs=args, inputkwargs=kwargs)
        return getattr(actions, name)(state)
    return return_action


def from_reaction(reaction: Reaction) -> Action:
    """
    Helper function that takes as parameter a reaction function
        and returns an equivalent action function.
    Used to chain actions and reactions together
        in an orchestration
    """
    def action_from_reaction(state: State) -> State:
        """
        This function incorporates the desired reaction
            as a closure
        """
        action, new_state = reaction(state)
        return action(new_state)
    return action_from_reaction


def next_event(event: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for reactions - changes the next event inside of the state
    Usage: @next_event('<event_name>')
    Internally replaces the next_event attribute of the state
        to the desired event
    """
    def decorator_next_event(reaction: Reaction) -> Reaction:
        @wraps(reaction)
        def wrapper(state: State) -> RxResp:
            state = state._replace(next_event=event)
            return reaction(state)
        return wrapper
    return decorator_next_event
