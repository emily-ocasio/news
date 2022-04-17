from turtle import end_fill
from actionutil import combine_actions, action2, RxResp, State
import calculations as calc
import choose

def article(state: State) -> RxResp:
    total = len(state.articles)
    row = state.articles[state.next_article]
    display, lines = calc.display_article(total, state.next_article, row, state.current_article_types)
    state = state._replace(article_lines = lines, remaining_lines = (len(lines)>35))
    return combine_actions(
        action2('clear_screen'),
        action2("print_message", message=display),
        choose.label(state)[0]
    ), state

def remaining_lines(state: State) -> RxResp:
    display = calc.display_remaining_lines(state.article_lines)
    state = state._replace(remaining_lines = False)
    return combine_actions(
        action2("print_message", message=display),
        choose.label(state)[0]
    ), state
 
def match_summary(state: State) -> RxResp:
    total = len(state.outputs)
    match_count = len(state.matches)
    nomatch_count = len(state.nomatches)
    msg = f"Matched = {match_count}, Not matched = {nomatch_count}, Total articles = {total}"
    return combine_actions(
        action2('print_message', message = msg),
        choose.match_group(state)[0]
    ), state

def location_count(state: State) -> RxResp:
    count_mass, count_nomass = calc.mass_divide(state.articles)
    msg = f"In Mass: {count_mass}, Not in Mass: {count_nomass}"
    return combine_actions(
        action2('print_message', message = msg), 
        choose.location(state)[0]
    ), state

def type_count(state: State) -> RxResp:
    count_good, count_bad = calc.type_divide(state.articles)
    msg = f"Good types: {count_good}, Bad types: {count_bad}"
    return combine_actions(
        action2('print_message', message = msg),
        choose.type(state)[0]
    ), state

def classify_progress(state: State) -> RxResp:
    msg = f"Number of articles: {len(state.articles)}"
    return action2('print_message', message = msg), state

def statistics(state: State) -> RxResp:
    msg = calc.statistic_summary(len(state.TP), len(state.TN), len(state.FP), len(state.FN))
    return action2('print_message', message= msg), state