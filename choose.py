from actionutil import combine_actions, action2, RxResp, State
import calculations as calc

initial_prompts = (
    "[R]eview matches from previous labels",
    "[F]ix errors by Record Id",
    "Enter [N]ew labels",
    "[A]uto categorize by date"
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
    "Review V[A]lidation2 dataset",
    "Review Te[S]t dataset",
    "Review T[E]st2 dataset",
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

def initial(state: State) -> RxResp:
    prompt, choices = calc.unified_prompt(initial_prompts)
    state = state._replace(next_event = "base_choice_made")
    return action2('get_user_input', prompt=prompt, choices=choices), state

def label_date(state: State) -> RxResp:
    prompt = "\n\nSelect date to label articles\nEnter date in format YYYYMMDD, [Q] to quit > "
    return action2('get_text_input', prompt=prompt), state

def label(state: State) -> RxResp:
    prompts = label_prompts + (('show e[X]tra lines',) if state.remaining_lines else tuple())
    allow_return = (state.article_kind == "review")
    prompt, choices = calc.unified_prompt(prompts, allow_return = allow_return)
    return action2('get_user_input', prompt=prompt, choices=choices, allow_return = allow_return), state

def choose_with_prompt(state: State, prompts, question) -> RxResp:
    prompt, choices = calc.unified_prompt(prompts)
    return combine_actions(
        action2('print_message', message = question),
        action2('get_user_input', prompt = prompt, choices = choices)
    ), state

def single_article(state: State) -> RxResp:
    msg = "Enter Record Id to fix, <Return> to go back, [Q] to quit > "
    return action2('get_text_input', prompt = msg), state

def dataset(state: State) -> RxResp:
    return choose_with_prompt(state, dataset_prompts, "Which dataset would you like to review?")

def review_label(state: State) -> RxResp:
    return choose_with_prompt(state, label_review_prompts, "Which label would you like to review?")

def match_group(state: State) -> RxResp:
    return choose_with_prompt(state, review_prompts, "Which group of articles would you like to review?")

def location(state: State) -> RxResp:
    return choose_with_prompt(state, location_prompts, "Which location group would you like to review")

def type(state: State) -> RxResp:
    return choose_with_prompt(state, type_prompts, "Which article types would you like to review?")

def dates_to_classify(state: State) -> RxResp:
    prompt = "Enter number of days to automatically classify > "
    return action2('get_number_input', prompt = prompt), state