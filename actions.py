"""
Actions (functions that produce side effects)
"""
import sqlite3
from sqlite3 import Cursor
#import os
from functools import wraps
from collections.abc import Callable
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
# from openai.chat.beta import ParsedChatCompletion
from secr_apis.gpt3_key import GPT3_API_KEY, GPT_API_KEY
from state import State, Action


db: sqlite3.Connection = sqlite3.connect('newarticles.db')
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
def print_message(message: str, end = '\n') -> None:
    """
    Display any message
    """
    print(message, end=end)


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
    #os.system('clear')
    return


@actiondef
def exit_program():
    """
    Display goodbye message and prepare to exit program
    """
    print("Good Bye!! To re-start program type 'python3 func.py'")
    db.close()


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
    cur: Cursor = db.execute(sql, args) #type: ignore
    rows = tuple(row for row in cur)
    return rows


@actiondef
def command_db(sql, **kwargs) -> None:
    """
    Execute SQL command
    If SQL has multiple statements separated by ;
        then apply parameters to each and execute one at a time
        and commit at the end
    """
    args = tuple(kwargs.values())
    # print(sql, kwargs)
    if ';' in sql:
        #print(sql, kwargs)
        sqls = sql.split(';')
        argcnts = tuple(ssql.count('?') for ssql in sqls)
        for i,ssql in enumerate(sqls):
            priorargs = sum(argcnts[:i])
            db.execute(ssql, args[priorargs:priorargs+argcnts[i]])
    else:
        #print(sql, kwargs)
        db.execute(sql,args)
    db.commit()


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

@actiondef
def get_number_range_input(prompt):
    """
    Prompt for user input and validate it is either a single integer or
        a range of numbers separated by a - (hyphen)
    Returns a tuple of one (single number) or two (range) integers
    """
    while True:
        answer = input(prompt)
        if answer.isnumeric():
            return (int(answer),)
        if answer.count('-') == 1:
            first, last = answer.split('-')
            if (first.isnumeric() and last.isnumeric()
                and int(first) <= int(last)):
                return (int(first), int(last))


@actiondef
def get_years_input(prompt):
    """
    Prompt for a year range and keep asking until single year or
        two years separated by a comma are entered
    Years must be in the range 1976-1984
    """
    def is_good_year(ans: str) -> bool:
        if not ans.isnumeric():
            return False
        return 1976 <= int(ans) <= 1984
    while True:
        answer = input(prompt)
        if answer == '':
            return answer
        years = answer.split(',')
        if len(years) == 1 and is_good_year(years[0]):
            return tuple(years)
        if len(years) == 2 and all(is_good_year(year) for year in years):
            return tuple(years)

@actiondef
def get_month_input(prompt):
    """
    Prompt for a specific Year-Month
    Must be between 1976-1984
    """
    def is_good_month(ans: str) -> bool:
        if len(ans) != 7 or ans[4] != '-':
            return False
        year= ans[0:4]
        month = ans[5:7]
        return (year.isnumeric() and month.isnumeric()
                    and (1976 <= int(year) <= 1984)
                    and (1 <= int(month) <= 12)
        )
    while True:
        answer = input(prompt)
        if is_good_month(answer):
            return answer

@actiondef
def prompt_gpt3(prompt, msg, model = 'davinci'):
    """
    Prompt GPT3 - use for older models only
    """
    models = {'curie': 'text-curie-001', 'davinci': 'text-davinci-003',}
    openai.api_key = GPT3_API_KEY
    response = openai.completions.create(
        model=models[model],
        # messages=[
        # {
        #     'role': 'user', 'content': prompt
        # }],
        prompt = prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text, msg
    # return response.choices[0].message.content, msg

@actiondef
def prompt_gpt(system, user, model = 'mini', max_tokens = 256,
               response_type = None) -> tuple:
    """
    Prompt GPT - newer API for GPT3.5 and later models
    When response_type is included, the beta Structured Output
        endpoint will be used
        response_type should be a pydantic type with expressive
        attribute names that will be used by the model to respond
    """
    models = {'mini': 'gpt-4o-mini', '4o': 'gpt-4o'}
    client = OpenAI(api_key=GPT_API_KEY)
    messages: list[ChatCompletionMessageParam] = [
        { 'role': 'system', 'content': system },
        { 'role': 'user', 'content': user }
    ]
    if response_type is not None:
        response_parse = client.beta.chat.completions.parse(
            model=models[model],
            seed = 42,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format = response_type)
        gpt_response = response_parse.choices[0].message.parsed
        usage = response_parse.usage
    else:
        response = client.chat.completions.create(
        model=models[model],
        seed = 42,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens)
        gpt_response = response.choices[0].message.content
        usage = response.usage

    total_tokens = 0 if usage is None else usage.total_tokens
    return gpt_response, system, total_tokens
