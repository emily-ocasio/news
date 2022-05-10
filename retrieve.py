from actionutil import combine_actions, action2, RxResp, Reaction, State
from typing import Callable
import calculations as calc
import functools

def query(query_type: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for queries - changes the next event inside of the state
    Usage: @query('<query_type>')
    Internally replaces the next_event attribute of the state to "query_retrieved"
        and sets the "query_type" attribute 
    """
    def decorator_query(c: Reaction) -> Reaction:
        @functools.wraps(c)
        def wrapper(state: State) -> RxResp:
            state = state._replace(next_event = 'query_retrieved', query_type = query_type)
            return c(state)
        return wrapper
    return decorator_query

@query('articles')
def unverified(state: State) -> RxResp:
    countsql, sql = calc.unverified_articles_sql()
    return action2('query_db', sql=sql, date=state.article_date), state

@query('matches')
def verified(state: State) -> RxResp:
    sql = calc.verified_articles_sql()
    return action2('query_db', sql, label=state.review_label, dataset = state.review_dataset), state

@query('all')
def all(state: State) -> RxResp:
    sql = calc.all_articles_sql()
    return action2('query_db', sql, dataset = state.review_dataset), state

@query('article_types')
def article_types(state: State) -> RxResp:
    sql = calc.retrieve_types_sql()
    row = state.articles[state.next_article]
    return action2('query_db', sql=sql, recordid = row['RecordId']), state

@query('unclassified_articles')
def to_auto_classify(state: State) -> RxResp:
    sql = calc.articles_to_classify_sql()
    days = state.dates_to_classify
    return action2('query_db', sql=sql, days=days), state

@query('single_article')
def single_article(state: State) -> RxResp:
    sql = calc.single_article_sql()
    id = state.article_id
    return action2('query_db', sql = sql, id = id), state