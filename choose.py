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
    "Assign articles to [H]omicides"
)

label_prompts = (
    "[M]assachussetts homicides",
    "[O]ther location homicides",
    "[N]ot homicides",
    "[P]ass and label later",
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


@choice('initial')
def initial(state: State) -> RxResp:
    """
    Initial application menu
    """
    prompt, choices = calc.unified_prompt(initial_prompts)
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
    allow_return = (state.article_kind in ('review', 'assign', 'reclassify'))
    prompt, choices = calc.unified_prompt(prompts, allow_return=allow_return)
    return action2('get_user_input',
                    prompt=prompt, choices=choices, allow_return=allow_return
    ), state


def choose_with_prompt(state: State, prompts, question) -> RxResp:
    """
    Generic choice function based on prompt tuple
    """
    prompt, choices = calc.unified_prompt(prompts)
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
    return choose_with_prompt(state, review_prompts,
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
    """
    prompt = "Enter number of days to assign classification > "
    return action2('get_number_input', prompt=prompt), state


@choice('dates_to_reclassify')
def dates_to_reclassify(state: State) -> RxResp:
    """
    Choose how many days to reclasssify auto-classified articles
    """
    prompt = "Enter number of days to verify auto-classification > "
    return action2('get_number_input', prompt=prompt), state


@choice('homicide_month')
def homicide_month(state: State) -> RxResp:
    """
    Choose the particular month from which to assign homicides
    """
    current_month = calc.year_month_from_article(
                                state.articles[state.next_article])
    prompt = f"Enter homicide month (<Return> for {current_month}) > "
    return action2('get_text_input', prompt=prompt), state
