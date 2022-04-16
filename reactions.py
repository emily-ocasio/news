# from token import RARROW
from actions import *
from actionutil import action, combine_actions, action2, RxResp
from calculations import *
import choose
import controller
import retrieve

label_prompts = (
    "[M]assachussetts homicides",
    "[O]ther location homicides",
    "[N]ot homicides",
    "[P]ass and label later"
)

review_prompts = (
    "Review articles that [M]atch homicide keywords",
    "Review articles that do [N]ot match keywords",
    "[C]ontinue without reviewing"
)

dataset_prompts = (
    "Review [T]raining dataset",
    "Review [V]alidation dataset",
    "[C]ontinue without reviewing"
)

def end_program(state: State) -> RxResp:
    return action2('exit_program'), state
 
def retrieve_article_types(state: State) -> RxResp:
    state = state._replace(next_event="types_retrieved")
    return retrieve.article_types(state)

def select_single_article(state: State) -> RxResp:
    msg = "Enter Record Id to fix, <Return> to go back, [Q] to quit > "
    state = state._replace(next_event = 'single_article_selected')
    return action2('get_text_input', prompt = msg), state

def retrieve_single_article(state: State) -> RxResp:
    if state.outputs == "Q":
        return end_program(state)
    if state.outputs == "":
        return choose.initial(state)
    sql = single_article_sql()
    state = state._replace(next_event = 'matches_retrieved', article_kind = 'review', next_article = 0)
    return action2('query_db', sql, id = state.outputs), state
