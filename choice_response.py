from actionutil import combine_actions, action2, RxResp, State
from reactions import select_single_article
from typing import Callable
import controller
import choose
import calculations as calc

def check_defaults(c: Callable[[State, str],RxResp]) -> Callable[[State],RxResp]:
    """
    Decorator for checking for default responses
    common to many prompts (such as [Q]uit and [C]ontinue)
    before proceeding with other choices
    Usage:
        add @check_defaults as decorator before choice response function
        choice response function must have additional argument "choice"
        which will be filled automatically with the selected choice
    """
    def wrapper(state: State) -> RxResp:
        choice = state.outputs
        if choice == 'Q':
            return controller.end_program(state)
        if choice == 'C':
            return choose.initial(state)
        return c(state, choice)
    return wrapper

@check_defaults
def initial(state: State, choice) -> RxResp:
    """
    Main menu
    """
    if choice == "N":
        return controller.new_labels(state)
    if choice == "R":
        return controller.review_datasets(state)
    if choice == "F":
        return select_single_article(state)
    if choice == 'A':
        return controller.auto_classify(state)
    raise Exception("Choice not supported")

@check_defaults
def label_date(state: State, choice) -> RxResp:
    """
    Date to verify new labels has been provided
    """
    state = state._replace(article_date = choice)
    return controller.retrieve_unverified(state)

@check_defaults
def new_label(state: State, choice) -> RxResp:
    """
    Respond to user selection of new (or updated) label
    """
    if choice == "X":
        return controller.show_remaining_lines(state)
    if state.article_kind == "review" and choice == "":
        state = state._replace(next_article = state.next_article+1)
        return controller.next_article(state)
    state = state._replace(new_label = choice)
    return controller.save_label(state)

@check_defaults
def dataset(state: State, choice) -> RxResp:
    if choice == "T":
        dataset = 'TRAIN'
    elif choice == "V":
        dataset = 'VAL'
    elif choice == 'S':
        dataset = 'TEST'
    else:
        raise Exception('Unsupported dataset choice')
    state = state._replace(review_dataset = dataset)
    return controller.select_review_label(state)

@check_defaults
def review_label(state: State, choice) -> RxResp:
    state = state._replace(review_label = choice)
    return controller.retrieve_verified(state)

@check_defaults
def match(state: State, choice) -> RxResp:
    if choice == "M":
        state = state._replace(articles = state.matches, article_kind = "review")
        return controller.select_location(state)
    if choice == "N":
        state = state._replace(articles = state.nomatches, article_kind = "review")
        return controller.select_location(state)
    raise Exception("Choice not supported")

@check_defaults
def location(state: State, choice) -> RxResp:
    if choice == 'M':
        state = state._replace(articles = calc.located_articles(state.articles))
        return controller.select_article_type(state)
    if choice == 'N':
        state = state._replace(articles = calc.located_articles(state.articles, False))
        return controller.select_article_type(state)
    raise Exception("Choice not supported")

@check_defaults
def type(state: State, choice) -> RxResp:
    if choice == 'G':
        state = state._replace(articles = calc.filter_by_type(state.articles))
        return controller.first_article(state)
    if choice == 'B':
        state = state._replace(articles = calc.filter_by_type(state.articles, False))
        return controller.first_article(state)
    raise Exception("Choice not supported")

def dates_to_classify(state: State) -> RxResp:
    if state.outputs == 0:
        return choose.initial(state)
    state = state._replace(dates_to_classify = state.outputs)
    return controller.classify_by_date(state)