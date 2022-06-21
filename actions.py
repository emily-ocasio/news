"""
Actions (functions that produce side effects)
"""
import sqlite3
from sqlite3 import Cursor
import os
from functools import wraps
from collections.abc import Callable
from state import State, Action


db = sqlite3.connect('newarticles.db')
db.row_factory = sqlite3.Row


def actiondef(act: Callable) -> Action:
    """
    Decorator for actions
    It places all the inputarguments (positional and named)
        into the state attributes inputargs and inputkwargs
    At the end of the action, it places the return value
        into the outputs attribute of the state
    """
    @wraps(act)
    def wrapper(state: State):
        outputs = act(*state.inputargs, **state.inputkwargs)
        if outputs is not None:
            state = state._replace(outputs=outputs)
        return state
    return wrapper


@actiondef
def no_op() -> None:
    """
    Action that does nothing
    """
    return

@actiondef
def print_message(message: str) -> None:
    """
    Display any message
    """
    print(message)
    return


@actiondef
def wait_enter() -> None:
    """
    Wait for user to press return key to continue
    """
    input("Press <return> to continue...")
    return


@actiondef
def clear_screen() -> None:
    """
    Clear the terminal screen
    """
    os.system('clear')
    return


@actiondef
def exit_program():
    """
    Exit the program after displaying goodbye message
    """
    print("Good Bye!! To re-start program type 'python3 func.py'")
    db.close()
    exit()


@actiondef
def get_user_input(prompt, choices, allow_return=False):
    """
    prompts user and keeps asking until provided one of the choices
    """
    answer = ""
    while not answer in choices:
        answer = input(prompt)
        answer = answer.upper()
        if allow_return and answer == "":
            return ""
    return answer


@actiondef
def query_db(sql, **kwargs):
    """
    Execute query SQL statement and collect results
    """
    args = tuple(kwargs.values())
    cur: Cursor = db.execute(sql, args)
    rows = tuple(row for row in cur)
    return rows


@actiondef
def command_db(sql, **kwargs) -> None:
    """
    Execute SQL command
    """
    args = tuple(kwargs.values())
    db.execute(sql, args)
    db.commit()
    return


@actiondef
def get_text_input(prompt, all_upper = True):
    """
    getting input text
    """
    answer = input(prompt)
    if all_upper:
        answer = answer.upper()
    return answer


@actiondef
def get_number_input(prompt):
    """
    Prompt for user input and keep asking until a number is provided
    """
    answer = ""
    while not answer.isnumeric():
        answer = input(prompt)
    return answer
