"""
Main entry point for the application
Monadic version
"""
from typing import cast
import sqlite3
import duckdb
from openai import OpenAI
from pymonad import Run, run_reader, run_state, run_base_effect, run_except, \
    run_sql, run_openai, run_splink, Environment, Namespace, ErrorPayload, \
    REAL_DISPATCH, Left, Right, Either, Tuple, put_line, pure, GPTModel, DbBackend, \
    StateRegistry
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
    db_path = "newarticles.db"
    duckdb_path = "news.duckdb"
    sqlite_con = sqlite3.connect(db_path)
    sqlite_con.row_factory = sqlite3.Row
    duck_con = duckdb.connect(duckdb_path)
    try:
        duck_con.execute("INSTALL sqlite_scanner;")
        duck_con.execute("LOAD sqlite_scanner;")
        attached = {
            row[1]
            for row in duck_con.execute("PRAGMA database_list;").fetchall()
        }
        if "sqldb" not in attached:
            duck_con.execute(
                f"ATTACH '{db_path}' AS sqldb (TYPE SQLITE);"
            )
    except Exception:
        sqlite_con.close()
        duck_con.close()
        raise

    env: Environment = {
        "prompt_ns": Namespace(""),
        "prompts_by_ns": {},
        "connections": {
            DbBackend.SQLITE: sqlite_con,
            DbBackend.DUCKDB: duck_con,
        },
        "current_backend": DbBackend.SQLITE,
        "openai_default_model": GPTModel.GPT_5_NANO,
        "openai_models": {},
        "openai_client": lambda: OpenAI(api_key=GPT_API_KEY,
                                        timeout=20.0, max_retries=2),
        "fasttext_model": SentenceTransformerModel(),
        "mar_key": MAR_API_KEY,
        "extras": {}
    }
    try:
        prog = run_reader(env, run_splink(run_state(StateRegistry.from_state(
                AppState.mempty()), \
                run_except(initialize_program()))))
        run_result = run_base_effect(REAL_DISPATCH, prog)
        prev_state = run_result.fst
        main_trampoline(env, prev_state)
    finally:
        exit_program(env["connections"])

# assume: Run, run_reader, run_state, run_except, run_sql,
# run_openai, run_base_effect
# and your AppState dataclass plus intents (console/db/openai)

def build_tick(env: Environment, state0: StateRegistry[AppState])\
    -> Run[AfterTick]:
    """
    Returns Run[MainResult]
    MainResult is Tuple[StateRegistry[AppState], MainChoice]
    """
    tick = main_menu_tick()  # Run[NextAction] inside your controller layer

    # Choose semantics: preserve state even on error
    # (so you can inspect what happened)
    wrapped = run_reader(env,
                run_except(
                    run_sql(
                        run_openai(env["openai_client"],
                            run_splink(
                            run_state(state0, tick)
                        )
                    )
                )
            )
            )

    # Map Either to always produce AfterTick = (StateRegistry[AppState], NextStep),
    # deciding what to do on errors
    # Also runtime "cast" into AfterTick class
    def normalize(ei: Either[ErrorPayload, Tuple[StateRegistry[AppState], NextStep]]) \
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

def main_trampoline(env: Environment, state0: StateRegistry[AppState]) -> None:
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

def exit_program(connections) -> None:
    """
    Clean up resources and exit the program.
    """
    print("Exiting program...")
    # Perform any necessary cleanup here
    for con in connections.values():
        try:
            con.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
