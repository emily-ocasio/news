"""
Reactions that display information
"""
from actionutil import combine_actions, action2, next_event
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
    status = state.articles[state.next_article]['Status']
    status = '' if status is None else status
    if (state.article_kind == 'assign' or status in 'MP'):
        display += (
            f"\n{calc.article_notes(state.articles[state.next_article])}\n"
        )
    state = state._replace(
        next_event = 'main',
        article_lines=lines, remaining_lines=False
        if state.article_kind == 'assign'
        else(len(lines) > state.terminal_size[1] - 12))
    return combine_actions(
        action2('clear_screen'),
        action2("print_message", message=display),
    ), state


def article_extracts(state: State) -> RxResp:
    """
    Display extracts for that article
    Occurs when showing article specific to a homicide
    """
    display = calc.homicide_extract(state.articles[state.next_article])
    return action2('print_message', message = display), state


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


@next_event('main')
def classify_progress(state: State) -> RxResp:
    """
    Display how many left to be classified
    Only display at the beginning or in 100 intervals
    """
    total = len(state.articles)
    left = len(state.articles) - state.next_article
    if left == total or left % 100 == 0:
        msg = f"Articles left to classify: {left} out of {total}"
        return action2('print_message', message=msg), state
    return action2('no_op'), state


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
    Displays homicide tables (assigned and available) for a given article
    Occurs during article assignment
    """
    msg = (("No homicides assigned to article"
                if len(state.homicides_assigned) == 0
                else ("Homicides assigned to article:\n"
                    + calc.homicide_table(state.homicides_assigned)))
                    + calc.display_homicide_extracts(state.homicides_assigned)
            + "\nHomicides available to assign: "
            + (f"(Victim name contains '{state.homicide_victim}')\n"
                    if len(state.homicide_victim) > 0
                    else f"({(state.homicide_month)})\n")
            + ("No homicides found" if len(state.homicides) == 0
                else calc. homicide_table(state.homicides)))
    return action2('print_message', message=msg), state


def homicides(state: State) -> RxResp:
    """
    Display all homicides in a group
    Occurs when reviewing homicides to determine humanizing
    """
    msg = calc.homicide_table(state.homicides)
    return action2('print_message', message=msg), state

@next_event('start')
def candidate_victims(state: State) -> RxResp:
    """
    Display candidate victims with their record IDs and victim names.
    """
    victims = state.victims
    msg = "\n".join(
        f"Record {item.record_id} - Victim: {item.victim.victim_name}"
        for item in victims
    )
    state = state._replace(main_flow='initial')
    return action2('print_message', message=msg), state
