"""
Reactions to actions that collect user input
"""
from collections.abc import Callable
from state import RxResp, State, Reaction
import controller
import choose
import calculations as calc

class ChoiceException(Exception):
    """
    Exception for invalid choice
    """


def check_defaults(choice_reaction: Callable[[State, str], RxResp]) -> Reaction:
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
            state = state._replace(main_flow = 'start')
            return controller.main(state)
        return choice_reaction(state, choice)
    return wrapper


def respond(state: State) -> RxResp:
    """
    Dispatches another function in this module with the name equal to the
        value of state.choice_type
    """
    return globals()[state.choice_type](state)


def username(state: State) -> RxResp:
    """
    Respond to the intial signing with the user's name
    Used to add LastUpdated to all updates as a form of audit
    """
    state = state._replace(user = state.outputs
                            if len(state.outputs) >=5 else '')
    return controller.main(state)


@check_defaults
def initial(state: State, choice) -> RxResp:
    """
    Main menu
    """
    flow_choice = {
        'N': 'new_labels',
        'R': 'review',
        'F': 'single_review',
        'A': 'auto_categorize',
        'H': 'assign',
        'S': 'second_filter',
        'Z': 'humanize',
    }
    if choice not in flow_choice:
        raise ChoiceException("Choice not supported")
    # These options have not yet been migrated to the main controller flow
    if choice == "N":
        state = state._replace(article_kind = 'new')
        return controller.new_labels(state)
    if choice == "F":
        state = state._replace(article_kind = 'review')
        return controller.edit_single_article(state)
    if choice == 'A':
        return controller.auto_classify(state)
    if choice == "H":
        state = state._replace(article_kind = 'assign')
        return controller.assign_homicides(state)
    # The remaining options are part of the main controller flow
    state = state._replace(main_flow = flow_choice[choice])
    return controller.main(state)


@check_defaults
def label_date(state: State, choice) -> RxResp:
    """
    Date to select new labels (ground truth) has been provided
    """
    state = state._replace(article_date=choice)
    return controller.retrieve_unverified(state)


@check_defaults
def new_label(state: State, choice) -> RxResp:
    """
    Respond to user selection of new (or updated) label
    """
    if choice == "X":
        return controller.show_remaining_lines(state)
    if state.article_kind in (
            'review', 'assign', 'reclassify') and choice == "":
        return controller.increment_article(state)
    state = state._replace(new_label=choice)
    return controller.save_label(state)


@check_defaults
def dataset(state: State, choice) -> RxResp:
    """
    Respond to selection of desired dataset for review
    """
    dataset_choice = {
        'P': 'PASSED',
        'T': 'TRAIN',
        'V': 'VAL',
        'L': 'VAL2',
        'S': 'TEST',
        'E': 'TEST2',
        'A': 'CLASS',
        '1': '1',
        '2': '2',
    }
    if choice not in dataset_choice:
        raise ChoiceException("Unsupported dataset choice")
    # if choice == 'P':
    #     return controller.review_passed_articles(state)
    # if choice == "T":
    #     review_dataset = 'TRAIN'
    # elif choice == "V":
    #     review_dataset = 'VAL'
    # elif choice == "L":
    #     review_dataset = 'VAL2'
    # elif choice == 'S':
    #     review_dataset = 'TEST'
    # elif choice == "E":
    #     review_dataset = 'TEST2'
    # elif choice == 'A':
    #     state = state._replace(article_kind='reclassify')
    #     review_dataset = 'CLASS'
    # else:
    #     raise Exception('Unsupported dataset choice')
    # state = state._replace(review_dataset=review_dataset)
    #return controller.select_review_label(state)
    state = state._replace(review_type = dataset_choice[choice])
    return controller.main(state)


@check_defaults
def review_label(state: State, choice) -> RxResp:
    """
    Respond to selection of desired label
    """
    state = state._replace(review_label=choice)
    return controller.retrieve_verified(state)


@check_defaults
def match(state: State, choice) -> RxResp:
    """
    Respond to choice whether to review matched or unmatched records
    """
    if choice == "M":
        matches = state.matches
    elif choice == "N":
        matches = state.nomatches
    else:
        raise ChoiceException("Choice not supported")
    state = state._replace(articles=matches)
    return controller.select_location(state)


@check_defaults
def location(state: State, choice) -> RxResp:
    """
    Respond to choice whether to review Mass or non-Mass articles
    """
    if choice not in 'MN':
        raise ChoiceException("Choice not supported")
    state = state._replace(
        articles=calc.located_articles(state.articles, choice == 'M')
    )
    return controller.select_article_type(state)


@check_defaults
def article_type(state: State, choice) -> RxResp:
    """
    Respond to choice whether to review articles of good or bad type
    """
    if choice not in 'GB':
        raise ChoiceException("Choice not supported")
    state = state._replace(
        articles=calc.filter_by_type(state.articles, choice == 'G')
    )
    return controller.first_article(state)


@check_defaults
def single_article(state: State, choice) -> RxResp:
    """
    Respond to user entry of record id of single article to review
    """
    if choice == "":
        return choose.initial(state)
    state = state._replace(article_id=choice)
    return controller.retrieve_single_article(state)


def dates_to_classify(state: State) -> RxResp:
    """
    Respond to number of days to auto-classify
    """
    if state.outputs == 0:
        return choose.initial(state)
    state = state._replace(dates_to_classify=state.outputs)
    return controller.classify_by_date(state)


def dates_to_assign(state: State) -> RxResp:
    """
    Respond to number of days to assign reclassified articles
    """
    if state.outputs == 0:
        return choose.initial(state)
    state = state._replace(dates_to_assign=state.outputs)
    return controller.assign_by_date(state)


def years_to_assign(state: State) -> RxResp:
    """
    Respond to years to assign reclassified articles
    """
    if state.outputs == '':
        begin = 1976
        end = 1984
        state = state._replace(assign_begin='19760101', assign_end='19841231')
        return controller.assign_by_year(state)
    if len(state.outputs) == 1:
        begin = state.outputs[0]
        end = state.outputs[0]
    elif len(state.outputs) == 2:
        begin = state.outputs[0]
        end = state.outputs[1]
    else:
        raise ChoiceException('Incorrect year input')
    state = state._replace(assign_begin=f"{begin}0101", assign_end=f"{end}1231")
    return choose.months_to_assign(state)


def months_to_assign(state: State) -> RxResp:
    """
    Respond to months to assign reclassified articles
    """
    month = int(state.outputs)
    if month == 0:
        return controller.assign_by_year(state)
    if month < 1 or month > 12:
        return choose.years_to_assign(state)
    begin = state.assign_begin[0:4] + f"{month:0>2}" + '01'
    end = state.assign_end[0:4] + f"{month:0>2}" + '31'
    state = state._replace(assign_begin = begin, assign_end = end)
    return controller.assign_by_year(state)


def dates_to_reclassify(state: State) -> RxResp:
    """
    Respond to number of days to reclassify auto-classified articles
    """
    if state.outputs == 0:
        return choose.initial(state)
    state = state._replace(dates_to_reclassify=state.outputs)
    return controller.reclassify_by_date(state)


def years_to_reclassify(state: State) -> RxResp:
    """
    Respond to years to reclassify articles
    """
    if state.outputs == '':
        begin = 1976
        end = 1984
    elif len(state.outputs) == 1:
        begin = state.outputs[0]
        end = state.outputs[0]
    elif len(state.outputs) == 2:
        begin = state.outputs[0]
        end = state.outputs[1]
    else:
        raise ChoiceException("Incorrect year input")
    state = state._replace(reclassify_begin=f"{begin}0101",
                            reclassify_end=f"{end}1231")
    return controller.reclassify_by_year(state)

@check_defaults
def assign_choice(state: State, choice) -> RxResp:
    """
    Perform selected choice during assignment
    """
    if choice == 'A':
        return choose.assigment(state)
    if choice == 'U':
        return choose.unassignment(state)
    if choice == 'M':
        return choose.humanize(state)
    if choice == 'H':
        return choose.homicide_month(state)
    if choice == 'V':
        return choose.homicide_victim(state)
    if choice == 'Y':
        return choose.homicide_county(state)
    if choice == 'G':
        state = state._replace(pre_article_prompt = 'few-shot2',
                post_article_prompt = 'few-shot')
        if len(state.homicides_assigned) == 1:
            state = state._replace(selected_homicide = 0)
            return controller.gpt3_humanize(state)
        return choose.gpt3_humanize(state)
    if choice == 'X':
        state = state._replace(pre_article_prompt ='article',
                post_article_prompt = 'alsopast4')
        if len(state.homicides_assigned) == 1:
            state = state._replace(selected_homicide = 0)
            return controller.gpt3_extract(state)
        return choose.gpt3_extract(state)
    if choice == 'L':
        state = state._replace(pre_article_prompt ='extract',
                post_article_prompt ='rewrite4')
        if len(state.homicides_assigned) == 1:
            state = state._replace(selected_homicide = 0)
            return controller.gpt3_small_extract(state)
        return choose.gpt3_small_extract(state)
    if choice == 'S':
        return controller.increment_article(state)
    if choice in 'NOPM':
        state = state._replace(new_label = choice)
        return controller.save_label(state)
    if choice in 'ED':
        state = state._replace(new_label = choice)
        return controller.save_assign_status(state)
    if choice == 'T':
        return controller.choose_new_note(state)
    raise ChoiceException(f"Unsupported dataset choice <{choice}>")


def homicide_month(state: State) -> RxResp:
    """
    Respond to selected homicide month to display
    """
    state = state._replace(homicide_month = state.outputs,
                            homicide_victim = '',
                            county = '')
    return controller.show_article(state)


def homicide_victim(state: State) -> RxResp:
    """
    Respond to desired name of victim to search for
    """
    state = state._replace(homicide_victim = state.outputs, county = '')
    return controller.show_article(state)


def homicide_county(state: State) -> RxResp:
    """
    Respond to desired name of county to search for
    """
    state = state._replace(county = state.outputs, homicide_victim = '')
    return controller.show_article(state)


def notes(state: State) -> RxResp:
    """
    Respond to newly entered notes for an article by user
    Occurs during assignment
    """
    if state.outputs == "":
        return controller.next_article(state)
    state = state._replace(new_notes = state.outputs)
    return controller.save_new_notes(state)


def assignments(state: State) -> RxResp:
    """
    Respond to selected homicide range for assignment
    If single number, ask for victim name
    """
    selection = state.outputs
    if selection[0] == 0 or selection[0] > len(state.homicides):
        return controller.next_article(state)
    if len(selection) == 1:
        state = state._replace(selected_homicides = (selection[0]-1,))
        return choose.victim(state)
    state = state._replace(victim = '', selected_homicides =
                tuple(h_ix-1 for h_ix in
                        range(selection[0],
                            min(selection[1]+1, len(state.homicides)+1))))
    return controller.save_assignment(state)


def humanize(state: State)  -> RxResp:
    """
    Respond to selected homicide for humanizing
        After selection, ask for humanizing level
    """
    selection = int(state.outputs)
    if selection == 0 or selection > len(state.homicides_assigned):
        return controller.next_article(state)
    state = state._replace(selected_homicide = selection-1)
    return choose.manual_humanizing(state)


def gpt3_humanize(state: State) -> RxResp:
    """
    Respond to selected homicide for GPT-3 to humanize
    """
    selection = int(state.outputs)
    if selection == 0 or selection > len(state.homicides_assigned):
        return controller.next_article(state)
    state = state._replace(selected_homicide = selection-1)
    return controller.gpt3_humanize(state)


def gpt3_extract(state: State) -> RxResp:
    """
    Respond to selected homicide for GPT-3 to extract
    """
    selection = int(state.outputs)
    if selection == 0 or selection > len(state.homicides_assigned):
        return controller.next_article(state)
    state = state._replace(selected_homicide = selection-1)
    return controller.gpt3_extract(state)


def gpt3_small_extract(state: State) -> RxResp:
    """
    Respond to selected homicide for GPT-3 to further extract
    """
    selection = int(state.outputs)
    if selection == 0 or selection > len(state.homicides_assigned):
        return controller.next_article(state)
    state = state._replace(selected_homicide = selection-1)
    return controller.gpt3_small_extract(state)


def victim(state: State) -> RxResp:
    """
    Respond to entered name of victim
    Occurs when homicide assignment is selected
    Try again if name is just one letter (not a space)
    """
    if len(state.outputs) == 1 and state.outputs != ' ':
        return choose.victim(state)
    state = state._replace(victim = state.outputs)
    return controller.save_assignment(state)


def unassignment(state: State) -> RxResp:
    """
    Respond to selected homicide number for unassignmnent
    """
    selected = int(state.outputs)
    if selected == 0 or selected > len(state.homicides_assigned):
        return controller.next_article(state)
    state = state._replace(selected_homicide = selected-1)
    return controller.save_unassignment(state)


def manual_humanizing(state: State) -> RxResp:
    """
    Respond to user choice of manual humanizing
    Occurs after user has selected the humanizing level
    """
    human = state.outputs
    if human =='' or not 1 <= int(human) <= 3:
        return controller.next_article(state)
    state = state._replace(humanizing = human)
    return controller.save_manual_humanizing(state)


def homicide_group(state: State) -> RxResp:
    """
    Respond to user entering homicide group in order to humanize
    """
    group = state.outputs
    if group in '0':
        state = state._replace(main_flow = 'start')
    state = state._replace(homicide_group = group)
    if int(group) > 5:
        state = state._replace(homicide_group = '')
    return controller.main(state)


@check_defaults
def humanize_action(state: State, choice) -> RxResp:
    """
    Respond to user enter humanizing action to perform
    """
    action_choice = {
        'M': 'manual',
        'A': 'auto',
        'C': 'end'
    }
    if choice not in action_choice:
        raise ChoiceException("Choice not supported")
    state = state._replace(homicide_action = action_choice[choice])
    return controller.main(state)


def homicide_to_humanize(state: State) -> RxResp:
    """
    Respond to homicide humber for manual humanizing
    """
    selected = int(state.outputs)
    if 0 < selected <= len(state.homicides):
        state = state._replace(current_homicide = selected-1)
    else:
        state = state._replace(homicide_action = '')
    return controller.main(state)


def humanize_homicide(state: State) -> RxResp:
    """
    Respond to user choice of manual humanizing
    Occurs after user has selected the humanizing level
    Specifically within the homicide groups
    """
    human = state.outputs
    if human =='' or not 0 <= int(human) <= 3:
        return controller.main(state)
    state = state._replace(humanizing = human)
    return controller.main(state)
