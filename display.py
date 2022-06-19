"""
Reactions that display information
"""
from actionutil import combine_actions, action2
from state import RxResp, State
import calculations as calc


def article(state: State) -> RxResp:
    """
    Display article content in full
    """
    total = len(state.articles)
    row = state.articles[state.next_article]
    display, lines = calc.display_article(total, state.next_article,
                                          row, state.current_article_types,
                                          limit_lines=0
                                          if state.article_kind == 'assign'
                                          else state.terminal_size[1])
    if state.article_kind == 'assign':
        display += (
            f"\n{calc.article_notes(state.articles[state.next_article])}\n"
        )
    state = state._replace(
        article_lines=lines, remaining_lines=False
        if state.article_kind == 'assign'
        else(len(lines) > state.terminal_size[1] - 12))
    return combine_actions(
        action2('clear_screen'),
        action2("print_message", message=display),
    ), state


def current_notes(state: State) -> RxResp:
    """
    Display current notes for article
    """
    msg = ("Current notes: \n"
           f"{calc.article_notes(state.articles[state.next_article])}\n")
    return action2('print_message', message=msg), state


def remaining_lines(state: State) -> RxResp:
    """
    Display additional lines for article which was too long
    """
    display = calc.display_remaining_lines(state.article_lines,
                                           limit_lines=state.terminal_size[1])
    state = state._replace(remaining_lines=False)
    return action2("print_message", message=display), state


def match_summary(state: State) -> RxResp:
    """
    Display summary counts of matched articles
    """
    total = len(state.outputs)
    match_count = len(state.matches)
    nomatch_count = len(state.nomatches)
    msg = (f"Matched = {match_count}, Not matched = {nomatch_count}, "
           f"Total articles = {total}"
           )
    return action2('print_message', message=msg), state


def location_count(state: State) -> RxResp:
    """
    Display summary count of article location
    """
    count_mass, count_nomass = calc.mass_divide(state.articles)
    msg = f"In Mass: {count_mass}, Not in Mass: {count_nomass}"
    return action2('print_message', message=msg), state


def type_count(state: State) -> RxResp:
    """
    Display summary counts of article types
    """
    count_good, count_bad = calc.type_divide(state.articles)
    msg = f"Good types: {count_good}, Bad types: {count_bad}"
    return action2('print_message', message=msg), state


def classify_progress(state: State) -> RxResp:
    """
    Display how many left to be classified
    """
    msg = f"Number of articles: {len(state.articles)}"
    return action2('print_message', message=msg), state


def statistics(state: State) -> RxResp:
    """
    Display full confusion matrix statistics
    """
    msg = calc.statistic_summary(
        len(state.TP),
        len(state.TN),
        len(state.FP),
        len(state.FN))
    return action2('print_message', message=msg), state


def homicide_table(state: State) -> RxResp:
    """
    Display formatted table of potential homicides
    """
    msg = calc.homicide_table(state.homicides)
    return action2('print_message', message=msg), state
