"""
Reactions that control overall program flow
"""
from actionutil import combine_actions, action2, from_reaction, next_event
from state import RxResp, State
import choose
import retrieve
import display
import save
import calculations as calc


def start_point(state: State) -> RxResp:
    """
    Initial starting point of application
    """
    return choose.initial(state)


def end_program(state: State) -> RxResp:
    """
    Final exit point of application
    """
    return action2('exit_program'), state


def new_labels(state: State) -> RxResp:
    """
    Labeling of articles by date
    """
    state = state._replace(article_kind='new', next_event='date_selected')
    return choose.label_date(state)


def retrieve_unverified(state: State) -> RxResp:
    """
    Query for unverified articles to label
    """
    return retrieve.unverified(state)


@next_event('start')
def no_articles(state: State) -> RxResp:
    """
    Endpoint when no articles are found by query
    """
    state = state._replace(article_kind='')
    return action2('print_message', message='No articles found.'), state


def first_article(state: State) -> RxResp:
    """
    Handle initial article of a group
    Could be articles to review or assign
    """
    if len(state.articles) == 0:
        return no_articles(state)
    state = state._replace(next_article=0)
    return next_article(state)


def next_article(state: State) -> RxResp:
    """
    Process next article of a group
    Could be article to review or assign
    First step is to retrieve the article types
    """
    if state.next_article >= len(state.articles):
        return last_article(state)
    if state.article_kind == 'assign':
        current_month = calc.year_month_from_article(
                                state.articles[state.next_article])
        state = state._replace(homicide_month = current_month)
    return retrieve.article_types(state)


@next_event('start')
def last_article(state: State) -> RxResp:
    """
    Process last article
    """
    return action2('print_message', message="All articles processed."), state


def show_article(state: State) -> RxResp:
    """
    Display article contents
    Preceeded by retrieving the article types
    If showing article in context of homicide assignment, first
        show table of homicides for the month
    """
    return combine_actions(
        from_reaction(display.article),
        from_reaction(retrieve.homicides_by_month
                        if state.article_kind == 'assign'
                        else choose.label)
    ), state


def show_remaining_lines(state: State) -> RxResp:
    """
    Display extra lines in long article
    """
    return combine_actions(
        from_reaction(display.remaining_lines),
        from_reaction(choose.label)
    ), state


def save_label(state: State) -> RxResp:
    """
    Save user provided label for article
    """
    return combine_actions(
        from_reaction(save.label),
        from_reaction(next_article)
    ), state

def save_assign_status(state: State) -> RxResp:
    """
    Save user selected assign status for article
    Comes from:
        User is prompted to assign an article
        and selects status such as "D" or "E"
    """
    return combine_actions(
        from_reaction(save.assign_status),
        from_reaction(next_article)
    ), state

def edit_single_article(state: State) -> RxResp:
    """
    Review label for single article
    """
    return choose.single_article(state)


def retrieve_single_article(state: State) -> RxResp:
    """
    Retrieve single article by Id for review
    """
    return retrieve.single_article(state)


def review_datasets(state: State) -> RxResp:
    """
    Select desired dataset for review
    """
    return choose.dataset(state)


def select_review_label(state: State) -> RxResp:
    """
    Select desired label subset to review
    """
    if state.review_dataset == 'CLASS':
        return choose.dates_to_reclassify(state)
    return choose.review_label(state)


def retrieve_verified(state: State) -> RxResp:
    """
    Retrieve label/verified articles for review
    """
    if state.review_label == 'A':
        return retrieve_all(state)
    msg = "Examining documents..."
    return combine_actions(
        action2('print_message', message=msg),
        from_reaction(retrieve.verified)
    ), state


def retrieve_all(state: State) -> RxResp:
    """
    Retrieve all articles in dataset for statistics
    """
    msg = "Computing statistics..."
    return combine_actions(
        action2('print_message', message=msg),
        from_reaction(retrieve.all_articles)
    ), state


@next_event('start')
def show_statistics(state: State) -> RxResp:
    """
    Display dataset statistics
    """
    return display.statistics(state)


def select_match_group(state: State) -> RxResp:
    """
    Select which match subset of articles to review
    """
    return combine_actions(
        from_reaction(display.match_summary),
        from_reaction(choose.match_group)
    ), state


def select_location(state: State) -> RxResp:
    """
    Select which location articles to review
    """
    return combine_actions(
        from_reaction(display.location_count),
        from_reaction(choose.location)
    ), state


def select_article_type(state: State) -> RxResp:
    """
    Select type of articles to review
    """
    return combine_actions(
        from_reaction(display.type_count),
        from_reaction(choose.article_type)
    ), state


def auto_classify(state: State) -> RxResp:
    """
    Automatically classify articles
    """
    return choose.dates_to_classify(state)


def classify_by_date(state: State) -> RxResp:
    """
    Retrieved articles to auto-classify by date
    """
    return retrieve.to_auto_classify(state)


def classify_articles(state: State) -> RxResp:
    """
    Begin classifying articles
    """
    if len(state.articles) == 0:
        return no_articles(state)
    return combine_actions(
        from_reaction(display.classify_progress),
        from_reaction(classify_next),
    ), state


def classify_next(state: State) -> RxResp:
    """
    Classify next article on list
    """
    if state.next_article >= len(state.articles):
        return all_classified(state)
    return save.classification(state)


@next_event('start')
def all_classified(state: State) -> RxResp:
    """
    Clean up dates database and notify user
        all articles in list have been classified
    """
    return combine_actions(
        from_reaction(save.dates_cleanup),
        action2('print_message', message="All articles classified.")), state


def assign_homicides(state: State) -> RxResp:
    """
    Assign homicides to articles
    First step when selected from main menu
    """
    return choose.dates_to_assign(state)


def assign_by_date(state: State) -> RxResp:
    """
    Retrieve articles to be assigned
    Occurs after user specifies how many dates to assign homicides
    """
    return retrieve.unassigned_articles(state)


def reclassify_by_date(state: State) -> RxResp:
    """
    Retrieve articles to be reclassified
    """
    return retrieve.auto_assigned_articles(state)


def select_homicide(state: State) -> RxResp:
    """
    Show homicides for month and select desired one
    """
    return retrieve.homicides_by_month(state)


@next_event('start')
def homicide_table(state: State) -> RxResp:
    """
    Display homicide table
    Preceeded by retrieval of homicide table
    Occurs while showing each article during assignment
    """
    return combine_actions(
        from_reaction(display.homicide_table),
        from_reaction(choose.assign_choice)
    ), state
