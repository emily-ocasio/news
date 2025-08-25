"""
Main entry point for the application
Monadic version
"""
from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    run_sqlite, \
    REAL_DISPATCH, put_line, view, Left, Either

from appstate import AppState, user_name
from get_username import get_username

def main_menu() -> Run[None]:
    """
    Display the main menu.
    """
    return \
        put_line("" \
            "Welcome to the Homicide Article Analysis System") ^ \
        get_username() ^ \
        view(user_name)>> (lambda user: \
        put_line(f"Hello, {user}! Let's get started. ")
        )

def main():
    """
    Main program that binds the intents and runs the Run monads.
    """
    env = {}
    prog = run_reader(env, run_state(AppState(), run_except(main_menu())))
    run_base_effect(REAL_DISPATCH, prog)

# assume: Run, run_reader, run_state, run_except, run_sqlite, run_openai, run_base_effect
# and your AppState dataclass plus intents (console/db/openai)

def build_tick(env: dict, state0: "AppState") -> "Run[tuple['AppState', str | None]]":
    """
    Returns Run[(new_state, next_action)]
    next_action: "quit" | None  (None means keep showing main menu)
    """
    tick = main_menu()  # Run[NextAction] inside your controller layer

    # Choose semantics: preserve state even on error (so you can inspect what happened)
    wrapped = run_reader(env,
              run_sqlite(env["db_path"],
              #run_openai(env["openai_client"],
              run_except(
                  run_state(state0, tick)   # => Run[Either e (AppState, NextAction)]
              )))

    # Map Either to always produce (AppState, action), deciding what to do on errors
    def normalize(ei: Either):
        if isinstance(ei, Left):
            # decide policy: keep prior state or use a safe recovery; here we keep prior
            return (state0, None)          # show an error in UI elsewhere if you like
        # Right((new_state, action))
        ns, action = ei.r
        return (ns, action)

    return wrapped.map(normalize)

def main_trampoline(env: dict, state0: "AppState") -> None:
    """"
    Main loop for the application.
    """
    state = state0
    while True:  # trampoline — no recursion, no TCO needed
        tick = build_tick(env, state)
        new_state, action = run_base_effect(REAL_DISPATCH, tick)
        state = new_state                       # <-- persist across ticks
        if action == "quit":
            break
        # else: loop back to main menu

if __name__ == "__main__":
    main()
