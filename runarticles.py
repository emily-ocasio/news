"""
Main entry point for the application
Monadic version
"""
from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    REAL_DISPATCH, put_line, get_line, set_, view

from appstate import AppState, user_name

def main_menu() -> Run[None]:
    """
    Display the main menu.
    """
    return \
        put_line("" \
            "Welcome to the Homicide Article Analysis System") >> (lambda _: \
        get_line("Please enter your name") >> (lambda name: \
        set_(user_name, name) >> (lambda _: \
        view(user_name) >> (lambda user: \
        put_line(f"Hello, {user}! Let's get started.")
        ))))
def main():
    """
    Main program that binds the intents and runs the Run monads.
    """
    env = {}
    prog = run_reader(env, run_state(AppState(), run_except(main_menu())))
    run_base_effect(REAL_DISPATCH, prog)

if __name__ == "__main__":
    main()
