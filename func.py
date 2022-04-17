from state import State
from reactions import *
import choose
import choice_response
import query_response
import controller

def what_next(state: State):
    dispatch = {
        'start': choose.initial,
        'base_choice_made': choice_response.initial,
        'date_selected': choice_response.label_date,
        'unverified_retrieved': query_response.articles,
        'types_retrieved': query_response.article_types,
        'label_selected': choice_response.new_label,
        'label_updated': controller.next_article,
        'review_dataset_selected': choice_response.dataset,
        'review_label_selected': choice_response.review_label,
        'matches_retrieved': query_response.matches,
        'all_retrieved': query_response.all,
        'statistics_displayed': choose.initial,
        'matches_processed': choice_response.match,
        'single_article_selected': choice_response.single_article,
        'single_article_retrieved': query_response.single_article,
        'separate_locations_displayed': choice_response.location,
        'article_type_selected': choice_response.type,
        'dates_to_classify_selected': choice_response.dates_to_classify,
        'auto_classify_retrieved': query_response.unclassified_articles,
        'ready_to_classify_next': controller.classify_next
    }
    return dispatch[state.next_event](state)

def main():
    """
    Main loop that controls the flow of the program
    Starting with an intial state, each pass of the loop does two things:
        1. Calls the what_next function which decides on the subsequent action
        2. Invokes the action returned why what_next
    Anything that has side effects (user input/output or database input/output)
        is an action
    Every other logic, either application flow or domain logic,
        including decisions on what to do, how to structure queries, etc
        are implemented as pure functionsr

    Some pure function return (but not invoke) functions that are actions.
        These actions are returned, along with a new immutable version of
        the application state, by the functions dispatched via what_next
    """
    current_state = State(next_event='start')
    while True:
        action, current_state = what_next(current_state)
        current_state = action(current_state)

if __name__ == "__main__":
    main();