"""
Functions that save data to database
"""
from itertools import chain
from actionutil import combine_actions, action2, next_event
from state import RxResp, State
import calculations as calc


def label(state: State) -> RxResp:
    """
    Save desired label for article
    """
    row = state.articles[state.next_article]
    sql = calc.verify_article_sql()
    return action2('command_db', sql=sql, status=state.new_label,
                   id=row['RecordId']), state


def assign_status(state: State) -> RxResp:
    """
    Save status of assignment process for article
    """
    row = state.articles[state.next_article]
    sql = calc.assign_status_sql()
    return action2('command_db', sql=sql, status=state.new_label,
                   id=row['RecordId']), state


def notes(state: State) -> RxResp:
    """
    Save user notes for an article
    """
    row = state.articles[state.next_article]
    sql = calc.update_note_sql()
    return action2('command_db', sql=sql, notes=state.new_notes,
                   id=row['RecordId']), state


def assignment(state: State) -> RxResp:
    """
    Assign a homicide to an article
    """
    shr_id = state.homicides[state.selected_homicide]['Id']
    record_id = state.articles[state.next_article]['RecordId']
    return (action2('command_db', sql=calc.assign_homicide_sql(),
                    shrid=shr_id,
                    recordid=record_id)
            if state.victim == ""
            else
            action2('command_db', sql=calc.assign_homicide_victim_sql(),
                        id=shr_id,
                        record=record_id,
                        victim=state.victim,
                        id2=shr_id)
            ), state


def assignments(state: State) -> RxResp:
    """
    Assign one or more homicides to an article
    If single homicide with new victim name is selected, use multi-statement
        SQL in a transaction
    For multiple homicides (no victims names allowed) then use multi-row
        insert SQL
    """
    record_id = state.articles[state.next_article]['RecordId']
    if state.victim != '':
        shr_id = state.homicides[state.selected_homicides[0]]['Id']
        return action2('command_db', sql=calc.assign_homicide_victim_sql(),
                        shrid = shr_id,
                        recordid = record_id), state
    shr_ids = tuple(state.homicides[hom_ix]['Id']
                        for hom_ix in state.selected_homicides)
    args = chain.from_iterable((shr_id, record_id) for shr_id in shr_ids)
    kwargs = {f"arg{i}":arg for i,arg in enumerate(args)}
    return action2('command_db', sql=calc.assign_homicide_sql(len(shr_ids)),
                                    **kwargs), state

def unassignment(state: State) -> RxResp:
    """
    Unassign a homicide to an article
    """
    sql = calc.unassign_homicide_sql()
    shr_id = state.homicides_assigned[state.selected_homicide]['Id']
    record_id = state.articles[state.next_article]['RecordId']
    return action2('command_db', sql=sql, shrid=shr_id,
                    recordid=record_id), state


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
    #state = state._replace(next_article=state.next_article+1)
    return combine_actions(action2('no_op') if auto_class == 'N'
                           else
                           action2('print_message', message=msg),
                           action2('command_db',
                                   sql=sql,
                                   auto_class=auto_class,
                                   id=row['RecordId'])), state


def dates_cleanup(state: State) -> RxResp:
    """
    Cleanup dates database by updating completion state
    """
    sql = calc.cleanup_sql()
    return action2('command_db', sql=sql), state
