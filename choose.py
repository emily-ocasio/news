"""
Reactions to determine prompts for user choices
"""
from functools import wraps
from collections.abc import Callable
from actionutil import combine_actions, action2
from state import RxResp, Reaction, State
import calculations as calc


def choice(choice_type: str) -> Callable[[Reaction], Reaction]:
    """
    Decorator for choices - changes the next event inside of the state
    Usage: @choice('<choice_type>')
    Internally replaces the next_event attribute of the state to "choice_made"
        and sets the "choice_type" attribute
    """
    def decorator_choice(reaction: Reaction) -> Reaction:
        @wraps(reaction)
        def wrapper(state: State) -> RxResp:
            state = state._replace(
                next_event='choice_made', choice_type=choice_type)
            return reaction(state)
        return wrapper
    return decorator_choice


initial_prompts = (
    "[R]eview matches from previous labels",
    "[F]ix errors by Record Id",
    "Enter [N]ew labels",
    "[A]uto categorize by date",
    "[S]econd filtering via GPT",
    "Assign articles to [H]omicides",
    "Determine humani[Z]ing coverage"
)

label_prompts = (
    "[M]assachussetts homicides",
    "[O]ther location homicides",
    "[N]ot homicides",
    "[P]ass and label later",
    "[C]continue without reviewing"
)

label_review_prompts = label_prompts + (
    "[A]ll labels (calculate statistics)",
)

review_prompts = (
    "Review articles that [M]atch homicide keywords",
    "Review articles that do [N]ot match keywords",
    "[C]ontinue without reviewing"
)

dataset_prompts = (
    "Review [T]raining dataset",
    "Review [V]alidation dataset",
    "Review Va[L]idation2 dataset",
    "Review Te[S]t dataset",
    "Review T[E]st2 dataset",
    "Review [A]uto classified articles",
    "Review previously [P]assed articles",
    "Review Assigned Group [1]",
    "Review Assigned Group [2]",
    "[C]ontinue without reviewing"
)

location_prompts = (
    "Review articles in [M]assachusetts",
    "Review articles [N]ot in Massachusetts",
    "[C]ontinue wihout reviewing"
)

type_prompts = (
    "Review articles with [G]ood article types",
    "Review articles with [B]ad (e.g. ads) article types",
    "[C]ontinue without reviewing"
)


assign_prompts = (
    "[A]ssign homicide by number",
    "[U]nassign homicide by number",
    "Hu[M]anize victim by number",
    "Send to [G]PT-3 for humanizing",
    "E[X]tract victim details for GPT-3 use",
    "Create sma[L]ler extract",
    "Change [H]omicide month to diaplay",
    "[S]kip article",
    "Homicide [E]arlier than 1976",
    "[D]one assigning",
    "Article [N]ot a homicide",
    "Homicide at [O]ther location",
    "Enter no[T]e",
    "[P]ass and review later",
    "Search for [V]ictim by name",
    "Display all homicides in a count[Y]",
    "[C]ontinue without assigning"
)


humanize_prompts = (
    "[M]anually determine humanizing",
    "[A]utomatically determine humanizing",
    "[C]ontinue without humanizing"
)


@choice('username')
def username(state: State) -> RxResp:
    """
    Ask for user's name at the beginning of the program
    Used to audit updates
    """
    msg = "Please enter your name > "
    return action2('get_text_input', prompt=msg, ), state


@choice('initial')
def initial(state: State) -> RxResp:
    """
    Initial application menu
    """
    prompt, choices = calc.unified_prompt(initial_prompts,
                                            width=state.terminal_size[0])
    return action2('get_user_input', prompt=prompt, choices=choices), state


@choice('label_date')
def label_date(state: State) -> RxResp:
    """
    Choose date to label articles
    """
    prompt = ("\n\nSelect date to label articles\n"
              "Enter date in format YYYYMMDD, [Q] to quit > "
              )
    return action2('get_text_input', prompt=prompt), state


@choice('new_label')
def label(state: State) -> RxResp:
    """
    Choose label for article
    """
    prompts = ((label_prompts[:-1] if state.article_kind == 'reclassify'
                else label_prompts)
               + (('show e[X]tra lines',) if state.remaining_lines
                  else tuple())
               )
    allow_return = state.article_kind in ('review', 'assign', 'reclassify')
    prompt, choices = calc.unified_prompt(prompts,
                                            allow_return=allow_return,
                                            width=state.terminal_size[0])
    return action2('get_user_input',
                   prompt=prompt, choices=choices, allow_return=allow_return
                   ), state


def choose_with_prompt(state: State, prompts, question) -> RxResp:
    """
    Generic choice function based on prompt tuple
    """
    prompt, choices = calc.unified_prompt(prompts, width=state.terminal_size[0])
    return combine_actions(
        action2('print_message', message=question),
        action2('get_user_input', prompt=prompt, choices=choices)
    ), state


@choice('single_article')
def single_article(state: State) -> RxResp:
    """
    Choose Record Id for single article review
    """
    msg = "Enter Record Id to fix, <Return> to go back, [Q] to quit > "
    return action2('get_text_input', prompt=msg), state


@choice('dataset')
def dataset(state: State) -> RxResp:
    """
    Choose dataset to review
    """
    return choose_with_prompt(state, dataset_prompts,
                              "Which dataset would you like to review?")


@choice('review_label')
def review_label(state: State) -> RxResp:
    """
    Choose label for review
    """
    return choose_with_prompt(state, label_review_prompts,
                              "Which label would you like to review?")


@choice('match')
def match_group(state: State) -> RxResp:
    """
    Choose whether to review matched or unmatched articles
    """
    return choose_with_prompt(
        state, review_prompts,
        "Which group of articles would you like to review?")


@choice('location')
def location(state: State) -> RxResp:
    """
    Choose whether to review Mass or non-Mass articles
    """
    return choose_with_prompt(state, location_prompts,
                              "Which location group would you like to review")


@choice('article_type')
def article_type(state: State) -> RxResp:
    """
    Choose whether to review article with Good or Bad types
    """
    return choose_with_prompt(state, type_prompts,
                              "Which article types would you like to review?")


@choice('dates_to_classify')
def dates_to_classify(state: State) -> RxResp:
    """
    Choose how many days to auto classify
    """
    prompt = "Enter number of days to automatically classify > "
    return action2('get_number_input', prompt=prompt), state


@choice('dates_to_assign')
def dates_to_assign(state: State) -> RxResp:
    """
    Choose how many days to assign homicides to articles
    This is first request after homicide assignment is selected
        via main menu
    """
    prompt = "Enter number of days to assign classification > "
    return action2('get_number_input', prompt=prompt), state


@choice('years_to_assign')
def years_to_assign(state: State) -> RxResp:
    """
    Choose which years to assign homicides to articles
    Useful when multiple users are assigning at the same time
    This is first request after homicide assignment is selected
        via main menu
    """
    prompt = ("Enter years to assign separated by comma (e,g. 1976,1978)\n"
              'or <Return> to select all years > ')
    return action2('get_years_input', prompt=prompt), state


@choice('months_to_assign')
def months_to_assign(state: State) -> RxResp:
    """
    Choose with particular month to assign homicide to articles
    This happens after a particular year is selected
    """
    prompt = ("Enter month to assign (number between 1 and 12)\n"
                "or '0' to select the entire year > ")
    return action2('get_number_input', prompt=prompt), state


@choice('dates_to_reclassify')
def dates_to_reclassify(state: State) -> RxResp:
    """
    Choose how many days to reclasssify auto-classified articles
    """
    prompt = "Enter number of days to verify auto-classification > "
    return action2('get_number_input', prompt=prompt), state


@choice('years_to_reclassify')
def years_to_reclassify(state: State) -> RxResp:
    """
    Choose which years to reclassify auto-classified articles
    Useful when multiple users are reviewing at the same time
    Occurs after Review Auto-Assigned articles is selected
    """
    prompt = ("Enter years to classify separated by comma (e.g. 1976,1978)\n"
              "or <Return> to select all years > ")
    return action2('get_years_input', prompt=prompt), state


@choice('assign_choice')
def assign_choice(state: State) -> RxResp:
    """
    Provide choices during assigment
    Occurs after article and list of homicides for the month is displayed
    Add prompt to make article 'M' if it's 'P' as part of assignment
    """
    prompts = assign_prompts + (("Article is [M]ass. homicide",)
                if state.articles[state.next_article]['Status'] == 'P'
                else tuple())
    return choose_with_prompt(state, prompts, "")


@choice('homicide_month')
def homicide_month(state: State) -> RxResp:
    """
    Choose new homicide month to show during assignment
    Occurs as a result of user choice
    """
    msg = "Enter new homicide month to display: "
    return action2('get_month_input', prompt=msg), state


@choice('homicide_victim')
def homicide_victim(state: State) -> RxResp:
    """
    Select the name of homicide victim during assignment
        in order to choose homicides to assign based on name
    """
    msg = "Enter name of victim to search for: "
    return action2('get_text_input', prompt=msg), state


@choice('homicide_county')
def homicide_county(state: State) -> RxResp:
    """
    Select the name of the county during assignment
        in order to choose homicides to assign based on county
    """
    msg = "Enter name of county to search for: "
    return action2('get_text_input', prompt=msg), state


@choice('notes')
def notes(state: State) -> RxResp:
    """
    Enter notes to be added to article
    Occurs during assignment after chooses to add note
    """
    msg = "Enter new note or <return> to leave unchanged > "
    return action2('get_text_input', prompt=msg, all_upper = False), state


@choice('assignments')
def assigment(state: State) -> RxResp:
    """
    Select row number of desired homicide to assign to current article
    """
    msg = "Select homicide number (n) to assign, 0 to go back > "
    return action2('get_number_range_input', prompt=msg), state


@choice('victim')
def victim(state: State) -> RxResp:
    """
    Select new victim name
    Occurs when a particular homicide is selected for assignment
    """
    current = state.homicides[state.selected_homicides[0]]['Victim']
    msg = (f"Enter victim's name (<Return> to keep [{current}]) > "
                if current
                else
                  "Enter victim's name (<Return> to keep as None) > ")
    return action2('get_text_input', prompt=msg, all_upper=False), state


@choice('unassignment')
def unassignment(state: State) -> RxResp:
    """
    Select row number of desired homicide to unassign (delete from
        list of assigned homicides for an article)
    """
    msg = "Select homicide number (k) to unassign, 0 to go back > "
    return action2('get_number_input', prompt=msg), state


@choice('humanize')
def humanize(state: State) -> RxResp:
    """
    Select row number of desired homicide to humanize
    """
    msg = "Select homicide number (k) to manually humanize, 0 to go back > "
    return action2('get_number_input', prompt=msg), state


@choice('manual_humanizing')
def manual_humanizing(state: State) -> RxResp:
    """
    Select humanizing level for human ground truth
    """
    msg = "Select humanizing level (1-3) > "
    return action2('get_number_input', prompt=msg), state


@choice('gpt3_humanize')
def gpt3_humanize(state: State) -> RxResp:
    """
    Select row number of desired homicide to sent to GPT-3 for humanizing
    """
    msg = ("Select homicide number (k) to send to GPT-3 for humanizing, "
            "or 0 to go back > ")
    return action2('get_number_input', prompt=msg), state


@choice('gpt3_extract')
def gpt3_extract(state: State) -> RxResp:
    """
    Select row number of desired homicide to extract for humanizing
    """
    msg = ("Select homicide number (k) to extract for humanizing, "
            "or 0 to go back > ")
    return action2('get_number_input', prompt=msg), state


@choice('gpt3_small_extract')
def gpt3_small_extract(state: State) -> RxResp:
    """
    Select row number of desired homicide to further extract for humanizing
    """
    msg = ("Select homicide number (k) to further extract for humanizing, "
            "or 0 to go back > ")
    return action2('get_number_input', prompt=msg), state


@choice('homicide_group')
def homicide_group(state: State) -> RxResp:
    """
    Select numbered homicide group for humanizing
    """
    msg = "Enter group number (1-5) or 0 to go back > "
    return action2('get_number_input', prompt=msg), state


@choice('humanize_action')
def humanize_action(state: State) -> RxResp:
    """
    Select the action to perform related to humanizing homicides
    """
    return choose_with_prompt(state, humanize_prompts,
                "Which action would you like to perform?")


@choice('homicide_to_humanize')
def homicide_to_humanize(state: State) -> RxResp:
    """
    Select which homicide from group to manually humanize
    """
    msg = "Enter homicide number (k) to manually humanize or 0 to go back > "
    return action2('get_number_input', prompt=msg), state


@choice('humanize_homicide')
def humanize_homicide(state: State) -> RxResp:
    """
    Select humanizing level for human ground truth
    """
    msg = "Select humanizing level (1-3) > "
    return action2('get_number_input', prompt=msg), state


@choice('articles_to_filter')
def articles_to_filter(state: State) -> RxResp:
    """
    Choose how many articles to filter
    """
    prompt = "Enter number of articles to filter > "
    return action2('get_number_input', prompt=prompt), state
