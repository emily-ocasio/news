"""
Reactions that control overall program flow
"""
from collections.abc import Callable
from actionutil import combine_actions, action2, from_reaction, next_event
from state import RxResp, State, Reaction
import choose
import gpt3_prompt
import retrieve
import display
import save
import calculations as calc

dispatch: dict[str, Reaction] = {}


class ControlException(Exception):
    """
    Custom exception for control flow
    """


def main_flow(flow: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for assigning main flow routes
    Usage:
        add @main_flow as decorator before controller reaction
        argument specifies the name of the main flow that is to be handled
    """
    def decorator_main(reaction: Reaction) -> Reaction:
        if flow in dispatch:
            raise ControlException(f"Main flow route {flow} duplicated")
        dispatch[flow] = reaction
        return reaction
    return decorator_main


def main(state: State) -> RxResp:
    """
    Central controller that dispatches based on main application workflow
    """
    if state.main_flow not in dispatch:
        raise ControlException(
            f"Main flow route {state.main_flow} not supported")
    return dispatch[state.main_flow](state)


@main_flow('start')
def start_point(state: State) -> RxResp:
    """
    Initial starting point of application
    """
    state = state._replace(next_step = 'begin',
                           article_kind='',
                           review_type='',
                           homicide_group='',
                           articles_to_filter=0,
                           articles=tuple(),
                           homicides=tuple())
    if len(state.user) == 0:
        return choose.username(state)
    return choose.initial(state)


@main_flow('end')
def end_program(state: State) -> RxResp:
    """
    Final exit point of application
    """
    state = state._replace(end_program=True)
    return action2('exit_program'), state


def new_labels(state: State) -> RxResp:
    """
    Labeling of articles by date
    """
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
    return action2('print_message', message='No articles found.'), state


def first_article(state: State) -> RxResp:
    """
    Handle initial article of a group
    Could be articles to review or assign
    """
    if len(state.articles) == 0:
        return no_articles(state)
    state = state._replace(next_article=-1)
    return increment_article(state)


def next_article(state: State) -> RxResp:
    """
    Process next article of a group
    Could be article to review or assign
    First step is to retrieve the article types
    """
    if state.next_article >= len(state.articles):
        return last_article(state)
    return retrieve.article_types(state)


def increment_article(state: State) -> RxResp:
    """
    Increment article point and process next article
    """
    state = state._replace(next_article=state.next_article+1)
    if (state.next_article < len(state.articles)
            and (state.article_kind == 'assign'
                 or (state.articles[state.next_article]['Status'] is not None
                and state.articles[state.next_article]['Status'] in 'MP'))):
        current_month = calc.year_month_from_article(
            state.articles[state.next_article])
        state = state._replace(homicide_month=current_month,
                               homicide_victim='', county='')
    return next_article(state)


@next_event('start')
def last_article(state: State) -> RxResp:
    """
    All articles have processed = print message and go back to main menu
    """
    return action2('print_message', message="All articles processed.\n"), state


def continue_retrieving_homicides(state: State) -> RxResp:
    """
    After homicides by article have been retrieved, proceed to
        retrieve the available homicides for assignment based on month
    Occurs when providing information for user for assignment
    If user has selected a victim name, retrieve based on name instead
    If user has selected a county, retrieve all county homicides instead
    """
    state = state._replace(next_step = 'show_homicide_table')
    if len(state.homicide_victim) > 0:
        # A victim name has been selected, so retrieve existing homicides
        # based on name
        return retrieve.homicides_by_victim(state)

    if len(state.county) > 0:
        # A county has been selected, retrieve based on county homicides
        return retrieve.homicides_by_county(state)

    # Retrieve based on month instead
    return retrieve.homicides_by_month(state)


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
        and proceed with next article
        except when making article 'M' during assignment
    """
    return combine_actions(
        from_reaction(save.label),
        from_reaction(refresh_article
                      if state.new_label == 'M'
                      and (state.article_kind == 'assign'
                           or state.articles[state.next_article]['Status']
                           in 'NMP')
                      else increment_article)
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
        from_reaction(increment_article)
    ), state


def save_new_notes(state: State) -> RxResp:
    """
    Save newly entered notes
    Comes from assignment after user has entered the desired notes
    """
    return combine_actions(
        from_reaction(save.notes),
        from_reaction(refresh_article)
    ), state


def save_assignment(state: State) -> RxResp:
    """
    Save newly selected assignment of homicide to an article
    Occurs after the user selects the corresponding number
    """
    return combine_actions(
        from_reaction(save.assignments),
        from_reaction(refresh_article)
    ), state


def save_unassignment(state: State) -> RxResp:
    """
    Delete requested previously assigned homicide
    Occurs after the user selects the homicide to unassign
    """
    return combine_actions(
        from_reaction(save.unassignment),
        from_reaction(refresh_article)
    ), state


def save_manual_humanizing(state: State) -> RxResp:
    """
    Save a manual humanizing level to a specific victim in an article
    """
    return combine_actions(
        from_reaction(save.manual_humanizing),
        from_reaction(refresh_article)
    ), state


def refresh_article(state: State) -> RxResp:
    """
    Refresh article after change was made
    Occurs within assignment, e.g. after new notes entered
    """
    return (retrieve.refreshed_article_with_extract(state)
            if state.main_flow == 'humanize'
            else retrieve.refreshed_article(state))


@main_flow('review')
def review_datasets(state: State) -> RxResp:
    """
    Main workflow for reviewing articles
    """
    rxn: Reaction | None = None
    next_step: str | None = None
    if state.articles_retrieved:
        state = state._replace(next_step = 'review_articles')

    match state.next_step:

        case 'begin':
            # First time entering review flow
            state = state._replace(
                article_kind='review',
                articles_retrieved=False)
            next_step = 'react_to_review_type'
            rxn = choose.dataset

        case 'react_to_review_type':
            # After selecting the type of article to review
            if state.review_type == 'PASSED':
                # Review previously passed articles
                next_step = 'retrieve_articles'
                rxn = retrieve.passed_articles
            elif state.review_type == 'SINGLE':
                # Review a single article
                next_step = 'retrieve_single_article'
                rxn = choose.single_article
            elif state.review_type == 'VICTIM_ID':
                # Review articles by victim id
                next_step = 'retrieve_by_victim_id'
                rxn = choose.victim_id
            else:
                state = state._replace(review_dataset=state.review_type)
                if state.review_dataset in '12':
                    # Review assigned articles by victim
                    next_step = 'review_articles'
                    rxn = retrieve.articles_humanizing_group
                elif state.review_type == 'CLASS':
                    # Review previously auto-classified articles to reclassify
                    state = state._replace(article_kind='reclassify')
                    next_step = 'retrieve_by_year'
                    rxn = choose.years_to_reclassify
                else:
                    # Continue with selection of label to review
                    next_step = 'retrieve_by_label'
                    rxn = choose.review_label

        case 'retrieve_single_article':
            next_step = 'review_articles'
            rxn = retrieve.single_article

        case 'retrieve_by_year':
            next_step = 'review_articles'
            rxn = retrieve.auto_assigned_articles_by_year

        case 'retrieve_by_label':
            # After selecting the label to review
            next_step = 'review_articles'
            rxn = retrieve.verified

        case 'retrieve_by_victim_id':
            next_step = 'review_articles'
            rxn = retrieve.articles_by_victim

        case 'review_articles':
            if len(state.articles) == 0:
                rxn = no_articles
            else:
                next_step = 'begin_article_review'
                rxn = review_articles

        case _:
            # Default case
            rxn = review_articles

    if next_step is not None:
        state = state._replace(next_step=next_step)
    return rxn(state)


def review_articles(state: State) -> RxResp:
    """
    Controller function to handle article review step.
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.next_step:
        case 'begin_article_review':
            # Start reviewing the next article.
            if state.next_article >= len(state.articles):
                # All articles have been reviewed.
                rxn = last_article
            else:
                article = state.articles[state.next_article]
                status = article['Status']
                if (state.article_kind == 'assign'
                    or (status is not None and status in 'MP')):
                    # Article should be assigned.
                    # Obtain the current month as the default
                    current_month = calc.year_month_from_article(article)
                    state = state._replace(homicide_month=current_month,
                                        homicide_victim='', county='')
                next_step = 'display_article'
                rxn = retrieve.article_types

        case 'display_article':
            # Article types have been retrieved and article is ready for display
            if (state.article_kind == 'assign'
                or state.articles[state.next_article]['Status'] == 'M'):
                next_step = 'retrieve_assignments'
            else:
                next_step = 'choose_new_label'
            rxn = display.article

        case 'retrieve_assignments':
            # Retrieve current assignments
            next_step = 'retrieve_homicide_table'
            rxn = retrieve.assigned_homicides_by_article

        case 'choose_new_label':
            # Choose a label for the article
            next_step = 'final_save'
            rxn = choose.label

        case 'retrieve_homicide_table':
            # Assignments have been retrieved
            rxn = continue_retrieving_homicides

        case 'show_homicide_table':
            # Potential homicidees habe been retrieved
            # Show possibiliies and choose
            rxn = homicide_table

        case 'assign_choice':
            # Assignment has been selected
            rxn = assign_choice

        case 'after_assignment' if len(state.selected_homicides) == 1:
            # Only one homicide selected - choose victim name first
            next_step = 'save_assignment'
            rxn = choose.victim

        case 'after_assignment' | 'save_assignment':
            next_step = 'display_article'
            rxn = save.assignments

        case 'save_unassignment':
            next_step = 'display_article'
            rxn = save.unassignment

        case 'save_notes':
            next_step = 'refresh_article'
            rxn = save.notes

        case 'save_victims':
            next_step = 'refresh_article'
            rxn = save.gpt_victims

        case next_step if next_step.startswith('gpt_filter'):
            rxn = filter_articles

        case 'choose_manual_humanizing':
            next_step = 'save_manual_humanizing'
            rxn = choose.manual_humanizing

        case 'save_manual_humanizing':
            next_step = 'display_article'
            rxn = save.manual_humanizing

        case 'final_save':
            # Process the user's input and save the label.
            if state.new_label == 'M':
                next_step = 'refresh_article'
            else:
                next_step = 'increment_article'
            rxn = save.label

        case 'refresh_article':
            next_step = 'display_article'
            rxn = retrieve.refreshed_article

        case 'increment_article':
            # Move to the next article.
            state = state._replace(next_article=state.next_article + 1,
                                   homicide_month = '')
            next_step = 'begin_article_review'
            rxn = review_articles

        # Handle unexpected current_step values.
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")

    if next_step is not None:
        state = state._replace(next_step=next_step)
    if rxn is not None:
        return rxn(state)


@main_flow('second_filter')
def second_filter(state: State) -> RxResp:
    """
    Second filter for articles using GPT-4 classification
    """
    next_step: str | None = None
    rxn : Reaction | None = None
    match state.next_step:
        case 'begin':
            next_step = 'retrieve_articles'
            rxn = choose.articles_to_filter
        case 'retrieve_articles':
            next_step = 'gpt_filter_begin'
            rxn = retrieve.articles_to_filter
        case 'next_article':
            state = state._replace(
                next_article=state.next_article + 1)
            next_step = 'gpt_filter_begin'
            rxn = filter_articles
        case step if step.startswith('gpt_filter'):
            rxn = filter_articles
        case 'refresh_article':
            # refresh article after classification so it can be reviewed later
            next_step = 'next_article'
            rxn = retrieve.refreshed_article
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")

    if next_step is not None:
        state = state._replace(next_step=next_step)
    if rxn is not None:
        return rxn(state)


def filter_articles(state: State) -> RxResp:
    """
    Filter articles one by one using GPT-4
    Can be called from second_filter or review main flows
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.next_step:
        case 'gpt_filter_begin' \
            if state.next_article >= len(state.articles):
            # Only invoked by second_filter main flow
            # No more articles to filter
            if len(state.articles) > 0:
                # All articles have been filtered, now retrieve them again
                # and then review the mismatches
                next_step = 'review_filtered_mismatches'
                rxn = review_filtered_mismatches
            else:
                # No articles to filter in the first place
                rxn = last_article
        case 'gpt_filter_begin' | 'gpt_filter_classify':
            state = state._replace(gpt_model = 'mini', gpt_max_tokens = 256,
                pre_article_prompt='homicide_type2',
                gpt3_action='classify_homicide')
            next_step = 'gpt_filter_check'
            rxn = gpt3_prompt.prompt_gpt4
        case 'gpt_filter_check' if state.gpt3_response == 'M':
            # Check location first because it's classified as homicide
            state = state._replace(
                pre_article_prompt='locationDC',
                gpt3_action='classify_location')
            next_step = 'gpt_filter_save'
            rxn = gpt3_prompt.prompt_gpt4
        case 'gpt_filter_check' | 'gpt_filter_save':
            # Either no need for location, or location was already determined
            # Return to the 'refrsh_article' step of the corresponding flow
            next_step = 'refresh_article'
            rxn = save.gpt_homicide_class
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")
    if next_step is not None:
        state = state._replace(next_step=next_step)
    if rxn is not None:
        return rxn(state)

def review_filtered_mismatches(state: State) -> RxResp:
    """
    Review filtered articles with mismatches
    Only include the articles that are not matched between GPT-4 and manual
    """
    state = state._replace(
        articles=tuple(filter(calc.gpt_manual_mismatch, state.articles)),
        main_flow='review',
        article_kind='review',
        next_article = 0,
        next_step='begin_article_review')
    return review_articles(state)


@main_flow('auto_classify')
def auto_classify_articles(state: State) -> RxResp:
    """
    Automatically classify articles
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.next_step:
        case 'begin':
            next_step = 'retrieve_articles'
            rxn = choose.dates_to_classify
        case 'retrieve_articles':
            next_step = 'classify_articles'
            rxn = retrieve.to_auto_classify
        case 'classify_articles' if len(state.articles) == 0:
            rxn = no_articles
        case 'classify_articles' if state.next_article >= len(state.articles):
            next_step = 'all_classified'
            rxn = save.dates_cleanup
        case 'classify_articles':
            next_step = 'save_classification'
            rxn = display.classify_progress
        case 'save_classification':
            next_step = 'next_article'
            rxn = save.classification
        case 'next_article':
            state = state._replace(next_article = state.next_article + 1)
            next_step = 'classify_articles'
            rxn = auto_classify_articles
        case 'all_classified':
            rxn = all_classified
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")

    if next_step is not None:
        state = state._replace(next_step=next_step)
    if rxn is not None:
        return rxn(state)

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

@next_event('start')
def all_classified(state: State) -> RxResp:
    """
    After all articles have been auto-classified, 
        notify user
    """
    return action2('print_message', message="All articles classified."), state


def assign_homicides(state: State) -> RxResp:
    """
    Assign homicides to articles
    First step when selected from main menu
    """
    return choose.years_to_assign(state)


def assign_by_date(state: State) -> RxResp:
    """
    Retrieve articles to be assigned
    Occurs after user specifies how many dates to assign homicides
    """
    return retrieve.unassigned_articles(state)


def assign_by_year(state: State) -> RxResp:
    """
    Retrieve articles to be assignned
    Occurs after user specifies which years to assign to
    """
    return retrieve.unassigned_articles_by_year(state)


def reclassify_by_date(state: State) -> RxResp:
    """
    Retrieve articles to be reclassified
    """
    return retrieve.auto_assigned_articles(state)


def reclassify_by_year(state: State) -> RxResp:
    """
    Retrieve articles to be reclassified by year
    Occurs after user enters years
    """
    return retrieve.auto_assigned_articles_by_year(state)


def select_homicide(state: State) -> RxResp:
    """
    Show homicides for month and select desired one
    """
    return retrieve.homicides_by_month(state)


def choose_new_note(state: State) -> RxResp:
    """
    Ask user for new article note in order to save it
    Occurs during assigment
    """
    return combine_actions(
        from_reaction(display.current_notes),
        from_reaction(choose.notes)
    ), state


def homicide_table(state: State) -> RxResp:
    """
    Display homicide table
    Preceeded by retrieval of homicide tables
    Occurs while showing each article during assignment
    """
    state = state._replace(next_step='assign_choice')
    return combine_actions(
        from_reaction(display.homicide_table),
        from_reaction(choose.assign_choice)
    ), state

def gpt3_humanize(state: State) -> RxResp:
    """
    Prompt GPT-3 to determine whether article is humanizing
    """
    state = state._replace(gpt3_action='humanize', gpt3_source='small')
    return gpt3_prompt.prompt_gpt(state)


def gpt3_extract(state: State) -> RxResp:
    """
    Prompt GPT-3 to extract the victim specific information
    """
    state = state._replace(gpt3_action='extract', gpt3_source='article')
    return gpt3_prompt.prompt_gpt(state)


def gpt3_small_extract(state: State) -> RxResp:
    """
    Prompt GPT-3 to further extract relevant victim specific information
    """
    state = state._replace(gpt3_action='small_extract', gpt3_source='extract')
    return gpt3_prompt.prompt_gpt(state)


def assign_choice(state: State) -> RxResp:
    """
    Respond to user assignment selection
    Occurs after showing homicide table
    """
    step : str | None = None
    match state.assign_choice:
        case 'assign':
            step = 'after_assignment'
            reaction = choose.assigment
        case 'unassign':
            step = "save_unassignment"
            reaction = choose.unassignment
        case 'humanize':
            step = "choose_manual_humanizing"
            reaction = choose.humanize
        case 'filter':
            step = 'gpt_filter_classify'
            reaction = filter_articles
        case 'victim_extract':
            step = 'save_victims'
            article = state.articles[state.next_article]
            prompt = calc.victim_extract_prompt_type(article)
            state = state._replace(gpt_model='4o', gpt_max_tokens=16000,
                                   pre_article_prompt=prompt,
                                   gpt3_action='victims')
            reaction = gpt3_prompt.prompt_gpt4
        case 'select_month':
            step = 'display_article'
            reaction =  choose.homicide_month
        case 'select_victim':
            step = 'display_article'
            reaction =  choose.homicide_victim
        case 'select_county':
            step = 'display_article'
            reaction =  choose.homicide_county
        case 'G':
            state = state._replace(pre_article_prompt = 'few-shot2',
                    post_article_prompt = 'few-shot')
            if len(state.homicides_assigned) == 1:
                state = state._replace(selected_homicide = 0)
                reaction = gpt3_humanize
            else:
                reaction = choose.gpt3_humanize
        case 'X':
            state = state._replace(pre_article_prompt ='article',
                    post_article_prompt = 'alsopast4')
            if len(state.homicides_assigned) == 1:
                state = state._replace(selected_homicide = 0)
                reaction = gpt3_extract
            else:
                reaction = choose.gpt3_extract
        case 'L':
            state = state._replace(pre_article_prompt ='extract',
                    post_article_prompt ='rewrite4')
            if len(state.homicides_assigned) == 1:
                state = state._replace(selected_homicide = 0)
                reaction = gpt3_small_extract
            else:
                reaction = choose.gpt3_small_extract
        case 'skip':
            # Skip to the next article
            step = 'increment_article'
            reaction = review_articles
        case 'N' | 'O' | 'P' | 'M' as label:
            state = state._replace(new_label = label)
            step = 'final_save'
            reaction = review_articles
        case 'E' | 'D' as label:
            # User selected new assigned statuses for this article
            state = state._replace(new_label = label)
            step = 'increment_article'
            reaction = save.assign_status
        case 'note':
            step = 'save_notes'
            reaction = choose_new_note
        case choice:
            raise ControlException(f"Unsupported assign choice <{choice}>")
    if step is not None:
        state = state._replace(next_step = step)
    return reaction(state)

@main_flow('humanize')
def humanize_homicides(state: State) -> RxResp:
    """
    Flow for determining whether articles are humanizing
        organized by homicide
    """
    # Always return to this flow as a default
    state = state._replace(next_event='main')
    if state.homicide_group == '':
        # First time - select homicide group to work with
        state = state._replace(homicides_retrieved=False,
                               homicide_action='')
        return choose.homicide_group(state)
    if not state.homicides_retrieved:
        # Group selected but not retrieved yet
        return retrieve.homicides_by_group(state)
    if len(state.homicides) == 0:
        # No homicides returned from query
        msg = "No homicides in that group"
        state = state._replace(main_flow='start')
        return action2('print_message', msg), state
    if state.homicide_action == '':
        state = state._replace(current_homicide=-1,
                               articles_retrieved=False,
                               article_kind='assign')
        return combine_actions(
            from_reaction(display.homicides),
            from_reaction(choose.humanize_action)
        ), state
    if state.homicide_action == 'manual':
        return humanize_homicides_manual(state)
    return humanize_homicides_auto(state)


def humanize_homicides_manual(state: State) -> RxResp:
    """
    Sub-flow for manual humanizing of the homicides by group
    Occurs when manual option is selected
    """
    if state.current_homicide == -1:
        state = state._replace(articles=tuple(),
                               humanizing='')
        return choose.homicide_to_humanize(state)
    if not state.articles_retrieved:
        return retrieve.articles_from_homicide(state)
    if state.next_article >= len(state.articles):
        state = state._replace(homicide_action='',
                               homicides_retrieved=False)
        msg = "All articles manually humanized.\n"
        return combine_actions(
            action2('print_message', msg),
            action2('wait_enter')
        ), state
    if state.humanizing == '0':
        state = state._replace(homicide_action='',
                               articles_retrieved=False,
                               homicides_retrieved=False)
        return action2('no_op'), state
    if state.humanizing == '' or int(state.humanizing) == 0:
        state = state._replace(humanizing_saved=False)
        msg = (f"\nHumanize article #{state.next_article+1} for victim "
               f"{state.homicides[state.current_homicide]['Victim']}")
        return combine_actions(
            action2('print_message', message=msg),
            from_reaction(display.article),
            from_reaction(display.article_extracts),
            from_reaction(choose.humanize_homicide)
        ), state
    if not state.humanizing_saved:
        # Manual humanizing level just selected, must save
        state = state._replace(humanizing_saved=True)
        return save.homicide_humanizing(state)
    # Humanzing level saved
    msg = f"Humanizing level saved ({state.humanizing})"
    state = state._replace(next_article=state.next_article+1,
                           humanizing='')
    return action2('print_message', message=msg), state


def humanize_homicides_auto(state: State) -> RxResp:
    """
    Sub-flow for automatically humanizing homicides in a group via GPT-3
    Occurs when auto function is selected
    """
    if state.current_homicide == -1:
        # First time - start auto humanizing
        state = state._replace(articles=tuple(),
                               humanizing='0',
                               humanizing_saved=False)
        state = state._replace(current_homicide=state.current_homicide+1)
    if state.current_homicide >= len(state.homicides):
        # Completed cycling through homicides
        msg = "All homicides humanized automatically"
        state = state._replace(homicide_action='',
                               homicides_retrieved=False)
        return action2('print_message', msg), state
    if not state.articles_retrieved:
        # Retrieve articles for this homicide
        return retrieve.articles_from_homicide(state)
    if (state.next_article >= len(state.articles)
            or state.homicides[state.current_homicide]['H'] == 3):
        # Auto humanizing completed for this homicide
        # (either because all articles analyzed or one article is humanizing)
        msg = "Auto-humanizing completed for "
        msg += f"#{state.current_homicide+1}/{len(state.homicides)}, "
        msg += f"victim: {state.homicides[state.current_homicide]['Victim']}, "
        msg += f"Id: {state.homicides[state.current_homicide]['Id']}, "
        msg += f"Number of articles: {len(state.articles)}, "
        msg += f"Humanizing = ({state.homicides[state.current_homicide]['H']})"
        state = state._replace(current_homicide=state.current_homicide+1,
                               articles=tuple(),
                               humanizing='0',
                               humanizing_saved=False,
                               articles_retrieved=False)
        return action2('print_message', message=msg), state
    return humanize_homicides_auto_gpt3(state)


def humanize_homicides_auto_gpt3(state: State) -> RxResp:
    """
    Sub-flow for automatically humanizing homicides
    This includes the portion that calls out GPT-3
    """
    if not state.articles[state.next_article]['Extract']:
        # Primary extract not yet created - create now
        if state.refresh_article:
            return refresh_article(state)
        state = state._replace(pre_article_prompt='article',
                               post_article_prompt='alsopast4',
                               gpt3_action='extract',
                               gpt3_source='article')
        return gpt3_prompt.prompt_gpt(state)
    if not state.articles[state.next_article]['SmallExtract']:
        # Secondary "small" extract not yet created - create it now
        if state.refresh_article:
            return refresh_article(state)
        state = state._replace(pre_article_prompt='extract',
                               post_article_prompt='rewrite4',
                               gpt3_action='small_extract',
                               gpt3_source='extract')
        return gpt3_prompt.prompt_gpt(state)
    if not state.articles[state.next_article]['Human']:
        # Final GPT-3 Humanizing not yet done - do it now
        if state.refresh_article:
            return refresh_article(state)
        state = state._replace(pre_article_prompt='few-shot2',
                               post_article_prompt='few-shot',
                               gpt3_action='humanize',
                               gpt3_source='small')
        return gpt3_prompt.prompt_gpt(state)
    # Auto humanizing complete
    state = state._replace(next_article=state.next_article+1)
    return retrieve.refreshed_homicide(state)


@main_flow('extract_victims')
def extract_victims(state: State) -> RxResp:
    """
    Flow for extracting victim information from articles.
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.next_step:
        case 'begin':
            next_step = 'retrieve_articles'
            rxn = choose.dataset
        case 'retrieve_articles':
            next_step = 'extract_victims'
            state = state._replace(review_dataset=state.review_type)
            rxn = retrieve.retrieve_articles_by_dataset
        case 'extract_victims':
            valid_victims = calc.gather_valid_generic_victims(state.articles)
            state = state._replace(victims=valid_victims)
            rxn = display.candidate_victims
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")

    if next_step is not None:
        state = state._replace(next_step=next_step)
    if rxn is not None:
        return rxn(state)
