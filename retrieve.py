"""
Reactions that culminate in databse query actions retrieving information
"""
import functools
from collections.abc import Callable
from actionutil import action2
from state import RxResp, Reaction, State
import calculations as calc


def query(query_type: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for queries - changes the next event inside of the state
    Usage: @query('<query_type>')
    Internally replaces the next_event attribute of the state to
        "query_retrieved" and sets the "query_type" attribute
    """
    def decorator_query(reaction: Reaction) -> Reaction:
        @functools.wraps(reaction)
        def wrapper(state: State) -> RxResp:
            state = state._replace(next_event='query_retrieved',
                                   query_type=query_type)
            return reaction(state)
        return wrapper
    return decorator_query


@query('many_articles')
def unverified(state: State) -> RxResp:
    """
    Retrieve unlabeled (unverified) articles
    """
    _, sql = calc.unverified_articles_sql()
    return action2('query_db', sql=sql, date=state.article_date), state


@query('verified')
def verified(state: State) -> RxResp:
    """
    Retrieve verified articles for review
    """
    sql = calc.verified_articles_sql()
    return action2('query_db', sql, label=state.review_label,
                   dataset=state.review_dataset), state


@query('all_articles')
def all_articles(state: State) -> RxResp:
    """
    Retrieve all articles from a dataset for full statistics
    """
    sql = calc.all_articles_sql()
    return action2('query_db', sql, dataset=state.review_dataset), state


@query('article_types')
def article_types(state: State) -> RxResp:
    """
    Retrieve list of article types for a specific article
    """
    sql = calc.retrieve_types_sql()
    row = state.articles[state.next_article]
    return action2('query_db', sql=sql, recordid=row['RecordId']), state


@query('unclassified_articles')
def to_auto_classify(state: State) -> RxResp:
    """
    Retrieve unclassified articles to auto-classify by date
    """
    sql = calc.articles_to_classify_sql()
    days = state.dates_to_classify
    return action2('query_db', sql=sql, days=days), state


@query('single_article')
def single_article(state: State) -> RxResp:
    """
    Retrieve single article for review by Record Id
    """
    sql = calc.single_article_sql()
    article_id = state.article_id
    return action2('query_db', sql=sql, id=article_id), state


@query('unassigned_articles')
def unassigned_articles(state: State) -> RxResp:
    """
    Retrieve unassigned auto-classifed articles ready to be assigned
    """
    sql = calc.articles_to_assign_sql()
    days = state.dates_to_assign
    return action2('query_db', sql=sql, days=days), state
