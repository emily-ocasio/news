from actionutil import combine_actions, action2, RxResp, State
import calculations as calc

def unverified(state: State) -> RxResp:
    countsql, sql = calc.unverified_articles_sql()
    return action2('query_db', sql=sql, date=state.article_date), state

def verified(state: State) -> RxResp:
    sql = calc.verified_articles_sql()
    return action2('query_db', sql, label=state.review_label, dataset = state.review_dataset), state

def article_types(state: State) -> RxResp:
    sql = calc.retrieve_types_sql()
    row = state.articles[state.next_article]
    return action2('query_db', sql=sql, recordid = row['RecordId']), state

def to_auto_classify(state: State) -> RxResp:
    sql = calc.articles_to_classify_sql()
    days = state.dates_to_classify
    return action2('query_db', sql=sql, days=days), state