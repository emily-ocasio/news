# from token import RARROW
from actions import *
from actionutil import action, combine_actions, action2, RxResp
from calculations import *
import choose
import retrieve

# def retrieve_article_types(state: State) -> RxResp:
#     state = state._replace(next_event="types_retrieved")
#     return retrieve.article_types(state)

# def retrieve_single_article(state: State) -> RxResp:
#     if state.outputs == "Q":
#         return end_program(state)
#     if state.outputs == "":
#         return choose.initial(state)
#     sql = single_article_sql()
#     state = state._replace(next_event = 'matches_retrieved', article_kind = 'review', next_article = 0)
#     return action2('query_db', sql, id = state.outputs), state
