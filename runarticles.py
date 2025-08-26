"""
Main entry point for the application
Monadic version
"""
from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    run_sqlite, Environment, Namespace, put_line, \
    REAL_DISPATCH, Left, Either, Tuple

from runinitial import initialize_program
from appstate import AppState

def main_menu() -> Run[None]:
    """
    Pre-trampoline program initialization
    """
    return \
        put_line("Good bye!")

def main() -> None:
    """
    Main program that binds the intents and runs the Run monads.
    """
    env: Environment = {
        "db_path": "newarticles.db",
        "prompt_ns": Namespace(""),
        "prompts_by_ns": {},
        "extras": {}
    }
    prog = run_reader(env, run_state(AppState.mempty(), \
            run_except(initialize_program())))
    run_result = run_base_effect(REAL_DISPATCH, prog)
    prev_state = run_result.fst
    main_trampoline(env, prev_state)

# assume: Run, run_reader, run_state, run_except, run_sqlite,
# run_openai, run_base_effect
# and your AppState dataclass plus intents (console/db/openai)

type TickResult = Tuple[AppState, str | None]

def build_tick(env: Environment, state0: AppState)\
    -> "Run[TickResult]":
    """
    Returns Run[(new_state, next_action)]
    next_action: "quit" | None  (None means keep showing main menu)
    """
    tick = main_menu()  # Run[NextAction] inside your controller layer

    # Choose semantics: preserve state even on error
    # (so you can inspect what happened)
    wrapped = run_reader(env,
              run_sqlite(env["db_path"],
              #run_openai(env["openai_client"],
              run_except(
                  run_state(state0, tick) # Run[Either e (AppState, NextAction)]
              )))

    # Map Either to always produce (AppState, action),
    # deciding what to do on errors
    def normalize(ei: Either) -> TickResult:
        if isinstance(ei, Left):
            # decide policy: keep prior state or use a safe recovery;
            # here we keep prior
            return Tuple(state0, None)    # show an error in UI elsewhere if you like
        # Right((Tuple(new_state, action)))
        #ns, action = ei.r
        return Tuple(ei.r.fst, ei.r.snd)

    return wrapped.map(normalize)

def main_trampoline(env: Environment, state0: "AppState") -> None:
    """"
    Main loop for the application.
    """
    state = state0
    while True:  # trampoline — no recursion, no TCO needed
        tick = build_tick(env, state)
        action_t = run_base_effect(REAL_DISPATCH, tick)
        state = action_t.fst                       # <-- persist across ticks
        if action_t.snd == "quit":
            break
        # else: loop back to main menu

if __name__ == "__main__":
    main()
