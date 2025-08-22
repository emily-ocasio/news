"""
Main entry point for the application
Monadic version
"""
from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    REAL_DISPATCH, put_line, view

from appstate import AppState, user_name
from get_username import get_username

def main_menu() -> Run[None]:
    """
    Display the main menu.
    """
    return \
        get_username() ^ \
        view(user_name) >> (lambda user2: \
        put_line("" \
            "Welcome to the Homicide Article Analysis System") ^ \
        get_username() ^ \
        view(user_name)>> (lambda user: \
        put_line(f"Hello, {user}! Let's get started. {user2}.")
        ))

def main():
    """
    Main program that binds the intents and runs the Run monads.
    """
    env = {}
    prog = run_reader(env, run_state(AppState(), run_except(main_menu())))
    run_base_effect(REAL_DISPATCH, prog)

if __name__ == "__main__":
    main()
