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
        raise ControlException("Main flow route not supported")
    return dispatch[state.main_flow](state)


@main_flow('start')
def start_point(state: State) -> RxResp:
    """
    Initial starting point of application
    """
    state = state._replace(current_step='not_started',
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
    Process last article
    """
    return action2('print_message', message="All articles processed.\n"), state


def show_article(state: State) -> RxResp:
    """
    Display article contents
    Preceeded by retrieving the article types
    If showing article in context of homicide assignment, first
        retrieve homicide information (assigned and available), starting
        with the already assigned homicides
    If article has murder (or pass) status, treat as assignment regardless
    """
    return combine_actions(
        from_reaction(display.article),
        from_reaction(retrieve.assigned_homicides_by_article
                      if (state.article_kind == 'assign'
                        or state.articles[state.next_article]['Status'] == 'M')
                      else choose.label)
    ), state


def continue_retrieving_homicides(state: State) -> RxResp:
    """
    After homicides by article have been retrieved, proceed to
        retrieve the available homicides for assignment based on month
    Occurs when providing information for user for assignment
    If user has selected a victim name, retrieve based on name instead
    If user has selected a county, retrieve all county homicides instead
    """
    state = state._replace(current_step = 'retrieve_homicides')
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


# def edit_single_article(state: State) -> RxResp:
#     """
#     Review label for single article
#     """
#     return choose.single_article(state)


# def retrieve_single_article(state: State) -> RxResp:
#     """
#     Retrieve single article by Id for review
#     """
#     return retrieve.single_article(state)


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
    last_step = 'retrieve_articles' if state.articles_retrieved \
        else state.current_step

    match last_step:
        case 'not_started':
            # First time entering review flow
            state = state._replace(
                article_kind='review',
                articles_retrieved=False)
            next_step = 'choose_review_type'
            rxn = choose.dataset

        case 'choose_review_type':
            # After selecting the type of article to review
            if state.review_type == 'PASSED':
                # Review previously passed articles
                next_step = 'retrieve_articles'
                rxn = retrieve.passed_articles
            elif state.review_type == 'SINGLE':
                # Review a single article
                next_step = 'choose_single_article'
                rxn = choose.single_article
            else:
                state = state._replace(review_dataset=state.review_type)
                if state.review_dataset in '12':
                    # Review assigned articles by victim
                    next_step = 'retrieve_articles'
                    rxn = retrieve.articles_humanizing_group
                elif state.review_type == 'CLASS':
                    # Review previously auto-classified articles to reclassify
                    state = state._replace(article_kind='reclassify')
                    next_step = 'choose_years_to_reclassify'
                    rxn = choose.years_to_reclassify
                else:
                    # Continue with selection of label to review
                    next_step = 'choose_label'
                    rxn = choose.review_label

        case 'choose_single_article':
            next_step = 'retrieve_articles'
            rxn = retrieve.single_article

        case 'choose_label':
            # After selecting the label to review
            next_step = 'retrieve_articles'
            rxn = retrieve.verified

        case 'retrieve_articles':
            if len(state.articles) == 0:
                rxn = no_articles
            else:
                next_step = 'next_article'
                rxn = review_articles

        case _:
            # Default case
            rxn = review_articles

    if next_step is not None:
        state = state._replace(current_step=next_step)
    return rxn(state)


def review_articles(state: State) -> RxResp:
    """
    Controller function to handle article review step.
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.current_step:
        case 'next_article':
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
                next_step = 'retrieve_article_types'
                rxn = retrieve.article_types

        case 'retrieve_article_types':
            # Article types have been retrieved and article is ready for display
            next_step = 'display_article'
            rxn = display.article

        case 'display_article':
            # Article was displayed
            if (state.article_kind == 'assign'
                or state.articles[state.next_article]['Status'] == 'M'):
                # Article can be assigned
                # Retrieve current assignments
                next_step = 'retrieve_assignments'
                rxn = retrieve.assigned_homicides_by_article
            else:
                # Article is ready for labeling - no assignment
                next_step = 'choose_label'
                rxn = choose.label

        case 'retrieve_assignments':
            # Assignments have been retrieved
            rxn = continue_retrieving_homicides

        case 'retrieve_homicides':
            # Potential homicidees habe been retrieved
            # Show possibiliies and choose
            rxn = homicide_table

        case 'choose_assign_option':
            # Assignment has been selected
            rxn = assign_choice

        case 'process_input':
            # Process the user's input and save the label.
            next_step = 'increment_article'
            rxn = save.label

        case 'increment_article':
            # Move to the next article.
            state = state._replace(next_article=state.next_article + 1)
            next_step = 'display_article'
            rxn = review_articles

        # Handle unexpected current_step values.
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")

    if next_step is not None:
        state = state._replace(current_step=next_step)
    if rxn is not None:
        return rxn(state)

# def review_passed_articles(state: State) -> RxResp:
#     """
#     Review previously passed articles
#     """
#     state = state._replace(article_kind = 'review')
#     return retrieve.passed_articles(state)


# def select_review_label(state: State) -> RxResp:
#     """
#     Select desired label subset to review
#     """
#     if state.review_dataset == 'CLASS':
#         return choose.years_to_reclassify(state)
#     return choose.review_label(state)

@main_flow('second_filter')
def second_filter(state: State) -> RxResp:
    """
    Second filter for articles using GPT-4 classification
    """
    next_step: str | None = None
    rxn : Reaction | None = None
    match state.current_step:
        case 'not_started':
            next_step = 'choose_number'
            rxn = choose.articles_to_filter
        case 'choose_number':
            # if state.articles_to_filter == 0:
            #     state = state._replace(main_flow='start')
            #     return main(state)
            next_step = 'retrieve_articles'
            rxn = retrieve.articles_to_filter
        case 'retrieve_articles':
            state = state._replace(
                pre_article_prompt='homicide_type',
                gpt3_action='classify_homicide')
            next_step = 'next_article'
        case 'save':
            state = state._replace(
                next_article=state.next_article + 1)
            next_step = 'next_article'

    if next_step is not None:
        state = state._replace(current_step=next_step)
    if rxn is not None:
        return rxn(state)

    return filter_articles(state)


def filter_articles(state: State) -> RxResp:
    """
    Filter articles one by one using GPT-4
    """
    next_step: str | None = None
    rxn: Reaction | None = None
    match state.current_step:
        case 'next_article':
            if state.next_article >= len(state.articles):
                rxn = last_article
            else:
                next_step = 'classify'
                rxn = gpt3_prompt.prompt_gpt4
        case 'classify':
            next_step = 'save'
            rxn = save.gpt_homicide_class
        case invalid:
            raise ControlException(f"Invalid current step: {invalid}")
    if next_step is not None:
        state = state._replace(current_step=next_step)
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


def increment_classify(state: State) -> RxResp:
    """
    Increment pointer to classify
    """
    state = state._replace(next_article=state.next_article+1)
    return classify_next(state)


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
    state = state._replace(current_step='choose_assign_option')
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
            step = 'choose_homicide'
            reaction = choose.assigment
        case 'U':
            reaction = choose.unassignment
        case 'M':
            reaction = choose.humanize
        case 'H':
            reaction =  choose.homicide_month
        case 'V':
            reaction =  choose.homicide_victim
        case 'Y':
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
            state = state._replace(next_article = state.next_article + 1,
                                current_step = 'next_article')
            reaction = review_articles
        case 'N' | 'O' | 'P' | 'M' as label:
            state = state._replace(new_label = label)
            reaction = save_label
        case 'E' | 'D' as label:
            state = state._replace(new_label = label)
            reaction = save_assign_status
        case 'T':
            reaction = choose_new_note
        case choice:
            raise ControlException(f"Unsupported assign choice <{choice}>")
    if step is not None:
        state = state._replace(current_step = step)
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
