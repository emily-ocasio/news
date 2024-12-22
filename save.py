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
                   user=state.user, id=row['RecordId']), state


def assign_status(state: State) -> RxResp:
    """
    Save status of assignment process for article
    """
    row = state.articles[state.next_article]
    sql = calc.assign_status_sql()
    return action2('command_db', sql=sql, status=state.new_label,
                   user=state.user, id=row['RecordId']), state


def notes(state: State) -> RxResp:
    """
    Save user notes for an article
    """
    row = state.articles[state.next_article]
    sql = calc.update_note_sql()
    return action2('command_db', sql=sql, notes=state.new_notes,
                   user=state.user, id=row['RecordId']), state


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
                       shrid=shr_id,
                       recordid=record_id,
                       user=state.user,
                       victim=state.victim,
                       id2=shr_id), state
    shr_ids = tuple(state.homicides[hom_ix]['Id']
                    for hom_ix in state.selected_homicides)
    args = chain.from_iterable((shr_id, record_id, state.user)
                               for shr_id in shr_ids)
    kwargs = {f"arg{i}": arg for i, arg in enumerate(args)}
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


def manual_humanizing(state: State) -> RxResp:
    """
    Assign a manual (human ground truth) humanizing level to a specific
        victim in an article
    """
    sql = calc.manual_humanizing_sql()
    shr_id = state.homicides_assigned[state.selected_homicide]['Id']
    record_id = state.articles[state.next_article]['RecordId']
    humanizing = state.humanizing
    return action2('command_db', sql=sql, human=humanizing, shrid=shr_id,
                    recordid=record_id), state


def gpt3_humanize(state: State) -> RxResp:
    """
    Assign a gpt3 humanizing level to a specific
        victim in an article
    """
    sql = calc.gpt3_humanizing_sql()
    homicide = (state.homicides[state.current_homicide]
                    if state.main_flow == 'humanize'
                    else state.homicides_assigned[state.selected_homicide])
    article = state.articles[state.next_article]
    shr_id = homicide['Id']
    record_id = article['RecordId']
    humanizing = state.humanizing
    human_manual = homicide['HM']
    pre_article = state.pre_article_prompt
    post_article = state.post_article_prompt
    prompt = state.gpt3_prompt
    response = state.gpt3_response
    return action2('command_db', sql=sql, human=humanizing, shrid=shr_id,
                    record_id=record_id, record_id2=record_id, shr_id2=shr_id,
                    human2=humanizing, human_manual=human_manual,
                    pre_article=pre_article, post_article=post_article,
                    prompt=prompt, response=response), state

def gpt_homicide_class(state: State) -> RxResp:
    """
    Assign a homicide class to an article based on GPT response
    """
    sql = calc.gpt_homicide_class_sql()
    article = state.articles[state.next_article]
    record_id = article['RecordId']
    gpt_class = state.gpt3_response
    state = state._replace(next_event='main')
    return action2('command_db', sql=sql, gptClass=gpt_class,
                   record_id=record_id), state

def gpt_victims(state: State) -> RxResp:
    """
    Save the extracted victims from an article
    """
    sql = calc.gpt_victims_sql()
    article = state.articles[state.next_article]
    record_id = article['RecordId']
    victims = state.gpt3_response
    return action2('command_db', sql=sql, victims=victims,
                   record_id=record_id), state

def gpt3_extract(state: State) -> RxResp:
    """
    Save extract of the article corresponding to the
        information specific to a particular homicide victim
    Extract could be initial or secondary
        (further extract removing standard data)
    """
    sql = (calc.gpt3_extract_sql()
            if state.gpt3_action == 'extract'
            else calc.gpt3_small_extract_sql())
    homicide = (state.homicides[state.current_homicide]
                if state.main_flow == 'humanize'
                else state.homicides_assigned[state.selected_homicide])
    article = state.articles[state.next_article]
    shr_id = homicide['Id']
    record_id = article['RecordId']
    pre_article = state.pre_article_prompt
    post_article = state.post_article_prompt
    prompt = state.gpt3_prompt
    response = state.gpt3_response
    extract = state.extract
    return action2('command_db', sql=sql, extract=extract, shrid=shr_id,
                    record_id=record_id, record_id2=record_id, shr_id2=shr_id,
                    human2='', human_manual='',
                    pre_article=pre_article, post_article=post_article,
                    prompt=prompt, response=response), state


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


def homicide_humanizing(state: State) -> RxResp:
    """
    Assign a manual (human ground truth) humanizing level to a specific
        victim in an article
    """
    sql = calc.manual_humanizing_sql()
    shr_id = state.homicides[state.current_homicide]['Id']
    record_id = state.articles[state.next_article]['RecordId']
    humanizing = state.humanizing
    return action2('command_db', sql=sql, human=humanizing, shrid=shr_id,
                    recordid=record_id), state
