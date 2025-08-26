"""
Monad for initial request for the username
"""
from pymonad import Run, get_line, set_, Prompt
from appstate import user_name

def get_username() -> Run[None]:
    """
    Get the username from the user.
    """
    return \
        get_line(Prompt("Please enter your name: ")) >> (lambda name: \
        set_(user_name, name) \
        )
