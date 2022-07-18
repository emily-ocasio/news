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
    Used during article processing as a prerequisite for displaying article
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


@query('refreshed_article')
def refreshed_article(state: State) -> RxResp:
    """
    Retrieve (refresh) current article for displaying again
    """
    sql = calc.single_article_sql()
    article_id = state.articles[state.next_article]['RecordId']
    return action2('query_db', sql=sql, id=article_id), state


@query('many_articles')
def passed_articles(state: State) -> RxResp:
    """
    Retrieve articles passed for further review
    """
    sql = calc.passed_articles_sql()
    return action2('query_db', sql=sql), state


@query('unassigned_articles')
def unassigned_articles(state: State) -> RxResp:
    """
    Retrieve unassigned articles post-review ready to be assigned
    User has selected the desired number of days
    """
    sql = calc.articles_to_assign_sql()
    days = state.dates_to_assign
    return action2('query_db', sql=sql, days=days), state


@query('unassigned_articles')
def unassigned_articles_by_year(state: State) -> RxResp:
    """
    Retrieve unassigned articles post-review ready to be assigned
        based on years
    User has selected the begin and end years
    """
    sql = calc.articles_to_assign_by_year_sql()
    begin = state.assign_begin
    end = state.assign_end
    return action2('query_db', sql=sql, begin=begin, end=end), state


@query('auto_assigned_articles')
def auto_assigned_articles(state: State) -> RxResp:
    """
    Retrieve auto-classified articles ready to be reclassified
    """
    sql = calc.articles_to_reclassify_sql()
    days = state.dates_to_reclassify
    return action2('query_db', sql=sql, days=days), state


@query('auto_assigned_articles')
def auto_assigned_articles_by_year(state: State) -> RxResp:
    """
    Retrieve auto-assigned articles for particular year(s)
    Occurs when user enter desired years to review auto-assigned articles
    """
    sql = calc.articles_to_reclassify_by_year_sql()
    begin = state.reclassify_begin
    end = state.reclassify_end
    return action2('query_db', sql=sql, begin=begin, end=end), state


@query('homicides_by_month')
def homicides_by_month(state: State) -> RxResp:
    """
    Retrieve homicides for a specific month
    """
    sql = calc.homicides_by_month_sql()
    month = state.homicide_month
    return action2('query_db', sql=sql, month=month), state


@query('homicides_by_month')
def homicides_by_victim(state: State) -> RxResp:
    """
    Retrieve homicides based on a match to victim name
    """
    sql = calc.homicides_by_victim_sql()
    victim = state.homicide_victim
    return action2('query_db', sql=sql, victim=victim), state


@query('homicides_by_month')
def homicides_by_county(state: State) -> RxResp:
    """
    Retrieve all homicides for a specific county
    """
    sql = calc.homicides_by_county_sql()
    county = state.county
    return action2('query_db', sql=sql, county=county), state


@query('assigned_homicides_by_article')
def assigned_homicides_by_article(state: State) -> RxResp:
    """
    Retrieve homicides already assigned to a particular article
    Needed to show user during assignment
    """
    sql = calc.homicides_assigned_by_article_sql()
    record_id = state.articles[state.next_article]['RecordId']
    return action2('query_db', sql=sql, id=record_id), state
