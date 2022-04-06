from typing import Callable
from state import State
import sqlite3
from sqlite3 import Cursor
import os

db = sqlite3.connect('newarticles.db')
db.row_factory = sqlite3.Row

# sets up decorator 
def actiondef(a: Callable) -> Callable[[State],State]:
    def wrapper(state: State):
        outputs = a(*state.inputargs, **state.inputkwargs)
        if outputs is not None:
            state = state._replace(outputs = outputs)
        return state
    return wrapper

# does nothing 
@actiondef
def no_op() -> None:
    return

# displaying any message 
@actiondef
def print_message(message: str) -> None:
    print(message)
    return

@actiondef
def wait_enter() -> None:
    input("Press <return> to continue...")
    return

@actiondef
def clear_screen() -> None:
    os.system('clear')
    return

@actiondef
def exit_program():
    print(f"Good Bye!! To re-start program type 'python3 func.py'")
    db.close()
    exit()

@actiondef
def get_user_input(prompt, choices, allow_return = False):
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
    args = tuple(kwargs.values())
    cur: Cursor = db.execute(sql, args)
    rows = tuple(row for row in cur)
    return rows

@actiondef
def command_db(sql, **kwargs) -> None:
    args = tuple(kwargs.values())
    db.execute(sql, args)
    db.commit()
    return

@actiondef
def get_text_input(prompt):
    """
    getting input text 
    """
    answer = input(prompt)
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
