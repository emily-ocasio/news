"""
Function for state initialization
"""

from rich import console
from state import State


def initialize_state(state: State) -> State:
    """
    Initialize state
    Determine size of terminal screen
    """
    cons = console.Console()
    state = state._replace(terminal_size=(cons.width, cons.height))
    return state
