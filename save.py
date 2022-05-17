"""
Functions that save data to database
"""
from actionutil import combine_actions, action2, next_event
from state import RxResp, State
import calculations as calc


def label(state: State) -> RxResp:
    """
    Save desired label for article
    """
    row = state.articles[state.next_article]
    sql = calc.verify_article_sql()
    state = state._replace(next_article=state.next_article+1)
    return action2('command_db', sql=sql, status=state.new_label,
                   id=row['RecordId']), state


@next_event('classified')
def classification(state: State) -> RxResp:
    """
    Save automatic classification
    """
    row = state.articles[state.next_article]
    sql = calc.classify_sql()
    auto_class = calc.classify(row)
    total = len(state.articles)
    msg = (
        f"Record: {row['RecordId']} (#{state.next_article} of {total}) "
        f"Date: {row['PubDate']}, classification: {auto_class}, "
        f"Title: {row['Title']} ")
    # if auto_class == 'M':
    #     disp, _ = calc.display_article(total, state.next_article, row, ())
    #     msg += f"\n" + disp
    state = state._replace(next_article=state.next_article+1)
    return combine_actions(action2('no_op') if auto_class == 'N'
                           else
                           action2('print_message', message=msg),
                           action2(
                               'command_db', sql=sql, auto_class=auto_class,
                               id=row['RecordId'])), state


def dates_cleanup(state: State) -> RxResp:
    """
    Cleanup dates database by updating completion state
    """
    sql = calc.cleanup_sql()
    return action2('command_db', sql=sql), state
