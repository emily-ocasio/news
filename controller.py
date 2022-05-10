from actionutil import combine_actions, action2, RxResp, State, from_reaction, next_event
import choose
import retrieve
import display
import save

def start_point(state: State) -> RxResp:
    return choose.initial(state)

def end_program(state: State) -> RxResp:
    return action2('exit_program'), state

def new_labels(state: State) -> RxResp:
    state = state._replace(article_kind = 'new', next_event = 'date_selected')
    return choose.label_date(state)

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
        return last_article(state)
    return retrieve.article_types(state)

@next_event('start')
def last_article(state: State) -> RxResp:
    return action2('print_message', message = "All articles processed."), state

def show_article(state: State) -> RxResp:
    return combine_actions(
        from_reaction(display.article),
        from_reaction(choose.label)
    ), state

def show_remaining_lines(state: State) -> RxResp:
    return combine_actions(
        from_reaction(display.remaining_lines),
        from_reaction(choose.label)
    ), state

def save_label(state: State) -> RxResp:
    return combine_actions(
        from_reaction(save.label),
        from_reaction(next_article)
    ), state

def edit_single_article(state: State) -> RxResp:
    return choose.single_article(state)

def retrieve_single_article(state: State) -> RxResp:
    return retrieve.single_article(state)

def review_datasets(state: State) -> RxResp:
    return choose.dataset(state)

def select_review_label(state: State) -> RxResp:
    return choose.review_label(state)

def retrieve_verified(state: State) -> RxResp:
    if state.review_label == 'A':
        return retrieve_all(state)
    msg = "Examining documents..."
    return combine_actions(
        action2('print_message', message = msg),
        from_reaction(retrieve.verified)
    ), state

def retrieve_all(state: State) -> RxResp:
    msg = "Computing statistics..."
    return combine_actions(
        action2('print_message', message = msg),
        from_reaction(retrieve.all)
    ), state

@next_event('start')
def show_statistics(state: State) -> RxResp:
    return display.statistics(state)

def select_match_group(state: State) -> RxResp:
    return combine_actions(
        from_reaction(display.match_summary),
        from_reaction(choose.match_group)
    ), state

def select_location(state: State) -> RxResp:
    return combine_actions(
        from_reaction(display.location_count),
        from_reaction(choose.location)
    ), state

def select_article_type(state: State) -> RxResp:
    return combine_actions(
        from_reaction(display.type_count),
        from_reaction(choose.type)
    ),state

def auto_classify(state: State) -> RxResp:
    return choose.dates_to_classify(state)

def classify_by_date(state: State) -> RxResp:
    return retrieve.to_auto_classify(state)

@next_event('start')
def assign_by_date(state:State) -> RxResp:
    return action2('print_message', message = "TODO..."), state

def classify_articles(state: State) -> RxResp:
    if len(state.articles) == 0:
        return no_articles(state)
    return combine_actions(
        from_reaction(display.classify_progress),
        from_reaction(classify_next),
    ), state

def classify_next(state: State) -> RxResp:
    if state.next_article >= len(state.articles): 
        return all_classified(state)
    return save.classification(state)

@next_event('start')
def all_classified(state: State) -> RxResp:
    return action2('print_message', message = "All articles classified."), state

def assign_homicides(state: State) -> RxResp:
    return choose.dates_to_assign(state)