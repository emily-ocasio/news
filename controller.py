from actionutil import combine_actions, action2, RxResp, State, Reaction
from typing import Callable
import choose
import retrieve
import display
import save
import functools

def next_event(event: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for reactions - changes the next event inside of the state
    Usage: @next_event('<event_name>')
    Internally replaces the next_event attribute of the state to the desired event 
    """
    def decorator_next_event(c: Reaction) -> Reaction:
        @functools.wraps(c)
        def wrapper(state: State) -> RxResp:
            state = state._replace(next_event = event)
            return c(state)
        return wrapper
    return decorator_next_event

def end_program(state: State) -> RxResp:
    return action2('exit_program'), state

def new_labels(state: State) -> RxResp:
    state = state._replace(article_kind = 'new', next_event = 'date_selected')
    return choose.label_date(state)

@next_event('unverified_retrieved')
def retrieve_unverified(state: State) -> RxResp:
    return retrieve.unverified(state)

@next_event('start')
def no_articles(state: State) -> RxResp:
    state = state._replace(article_kind = '')
    return action2('print_message', message = 'No articles found.'), state

def first_article(state: State) -> RxResp:
    if len(state.articles) == 0:
        return no_articles(state)
    state = state._replace(next_article = 0)
    return next_article(state)

def next_article(state: State) -> RxResp:
    if state.next_article >= len(state.articles):
        state = state._replace(next_event = 'start')
        return action2('print_message', message = "All articles processed."), state
    state = state._replace(next_event = 'types_retrieved')
    return retrieve.article_types(state)

@next_event('label_selected')
def show_article(state: State) -> RxResp:
    return display.article(state)

def show_remaining_lines(state: State) -> RxResp:
    return display.remaining_lines(state)

def save_label(state: State) -> RxResp:
    state = state._replace(next_event = 'label_updated')
    return save.label(state)

def edit_single_article(state: State) -> RxResp:
    state = state._replace(next_event = 'single_article_selected')
    return choose.single_article(state)

def retrieve_single_article(state: State) -> RxResp:
    state = state._replace(next_event = 'single_article_retrieved')
    return retrieve.single_article(state)

@next_event('review_dataset_selected')
def review_datasets(state: State) -> RxResp:
    return choose.dataset(state)

@next_event('review_label_selected')
def select_review_label(state: State) -> RxResp:
    return choose.review_label(state)

def retrieve_verified(state: State) -> RxResp:
    msg = "Examining documents..."
    state = state._replace(next_event = "all_retrieved" if state.review_label == 'A' else "matches_retrieved")
    return combine_actions(
        action2('print_message', message = msg),
        retrieve.all(state)[0] if state.review_label == 'A' else retrieve.verified(state)[0]
    ), state

@next_event('statistics_displayed')
def show_statistics(state: State) -> RxResp:
    return display.statistics(state)

def select_match_group(state: State) -> RxResp:
    state = state._replace(next_event = 'matches_processed')
    return display.match_summary(state)

def select_location(state: State) -> RxResp:
    state = state._replace(next_event = 'separate_locations_displayed')
    return display.location_count(state)

def select_article_type(state: State) -> RxResp:
    state = state._replace(next_event = 'article_type_selected')
    return display.type_count(state)

def auto_classify(state: State) -> RxResp:
    state = state._replace(next_event = 'dates_to_classify_selected')
    return choose.dates_to_classify(state)

def classify_by_date(state: State) -> RxResp:
    state = state._replace(next_event = 'auto_classify_retrieved')
    return retrieve.to_auto_classify(state)

def classify_articles(state: State) -> RxResp:
    if len(state.articles) == 0:
        return no_articles(state)
    state = state._replace(next_event = 'ready_to_classify_next')
    return display.classify_progress(state)

def classify_next(state: State) -> RxResp:
    if state.next_article >= len(state.articles):
        state = state._replace(next_event = 'start')
        return action2('print_message', message = "All articles classified."), state
    state = state._replace(next_event = 'ready_to_classify_next')
    return save.classification(state)
