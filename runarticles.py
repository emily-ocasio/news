"""
Main entry point for the application
Monadic version
"""
import sys
from typing import cast
from openai import OpenAI

from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    run_sqlite, run_openai, Environment, Namespace, ErrorPayload, \
    REAL_DISPATCH, Left, Right, Either, Tuple, put_line, pure, GPTModel

from article import ArticleAppError
from runinitial import initialize_program
from appstate import AppState
from mainmenu import main_menu_tick, AfterTick
from menuprompts import NextStep
from st_initialize import SentenceTransformerModel
from secr_apis.gpt3_key import GPT_API_KEY
from secr_apis.mar_key import MAR_API_KEY

def main() -> None:
    """
    Main program that binds the intents and runs the Run monads.
    """
    env: Environment = {
        "db_path": "newarticles.db",
        "duckdb_path": "news.duckdb",
        "prompt_ns": Namespace(""),
        "prompts_by_ns": {},
        "openai_default_model": GPTModel.GPT_5_NANO,
        "openai_models": {},
        "openai_client": lambda: OpenAI(api_key=GPT_API_KEY,
                                        timeout=20.0, max_retries=2),
        "fasttext_model": SentenceTransformerModel(),
        "mar_key": MAR_API_KEY,
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

def build_tick(env: Environment, state0: AppState)\
    -> Run[AfterTick]:
    """
    Returns Run[MainResult]
    MainResult is Tuple[AppState, MainChoice]
    """
    tick = main_menu_tick()  # Run[NextAction] inside your controller layer

    # Choose semantics: preserve state even on error
    # (so you can inspect what happened)
    wrapped = run_reader(env,
                run_except(
                run_sqlite(env['db_path'],
                run_openai(env["openai_client"],
                run_state(state0, tick) # Run[Either e (AppState, NextAction)]
                ))))

    # Map Either to always produce AfterTick = (AppState, NextStep),
    # deciding what to do on errors
    # Also runtime "cast" into AfterTick class
    def normalize(ei: Either[ErrorPayload, Tuple[AppState, NextStep]]) \
        -> Run[AfterTick]:
        match ei:
            case Left(err):
                # decide policy: keep prior state or use a safe recovery;
                # here we keep prior after showing error
                return \
                    (put_line(f"Error occurred: {err}\nTry again...") \
                    if err.app is None else \
                    put_line(cast(ArticleAppError, err.app).value)) ^ \
                    pure(AfterTick.make(state0, NextStep.CONTINUE))
            case Right(result):
                return pure(AfterTick.make(result.fst, result.snd))

    return wrapped >> normalize

def main_trampoline(env: Environment, state0: AppState) -> None:
    """
    Main loop for the application.
    """
    state = state0
    while True:  # trampoline — no recursion, no TCO needed
        tick = build_tick(env, state)
        after_tick = run_base_effect(REAL_DISPATCH, tick)
        state = after_tick.state     # <-- persist across ticks
        if after_tick.next_step == NextStep.QUIT:
            return
        # else: loop back to main menu

def exit_program() -> None:
    """
    Clean up resources and exit the program.
    """
    print("Exiting program...")
    # Perform any necessary cleanup here
    sys.exit()

if __name__ == "__main__":
    main()
    exit_program()
