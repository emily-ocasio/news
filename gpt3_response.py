"""
Reactions to handle the GPT-3 responses
"""
from state import RxResp, State
from actionutil import action2, combine_actions, from_reaction
import save
import controller
import calculations as calc


class GPTResponseException(Exception):
    """Exception for GPT response errors"""


def respond(state: State) -> RxResp:
    """
    Handle response from GPT
    """
    match state.gpt3_action:
        case 'classify_homicide':
            return respond_homicide_class(state)
        case 'classify_location':
            return respond_location_class(state)
    response, prompt = state.outputs
    msg = calc.prompt_response(prompt, response)
    state = state._replace(gpt3_prompt=prompt, gpt3_response=response,
                           refresh_article=True)
    if state.gpt3_action in ('extract', 'small_extract'):
        state = state._replace(extract=calc.remove_quotes(response))
        reaction = save.gpt3_extract
    elif state.gpt3_action == 'humanize':
        state = state._replace(
            humanizing=calc.humanizing_from_response(response, 'word'))
        reaction = save.gpt3_humanize
    else:
        raise GPTResponseException('Unknown GPT action')
    return combine_actions(
        action2('print_message', msg),
        from_reaction(reaction),
        action2('no_op' if state.main_flow == 'humanize' else 'wait_enter'),
        from_reaction(controller.main if state.main_flow == 'humanize'
                      else controller.refresh_article)
    ), state


def respond_homicide_class(state: State) -> RxResp:
    """
    Handle response from GPT for homicide classification
    """
    response, prompt = state.outputs
    response_text = response.classification.value
    response_code = calc.gpt_homicide_class_code(response.classification)
    msg = calc.prompt_response(prompt, "\n\n" + response_text)
    manual_class = state.articles[state.next_article]['Status']
    match = calc.is_gpt_homicide_class_correct(response_code, manual_class)
    record_id = state.articles[state.next_article]['RecordId']
    msg = "Manual class / Initial GPT class: " + \
            f"{manual_class} / {response_code}\n" + \
            f"Record Id: {record_id}\n\n"
    if response_code == 'M':
        msg = "Initial GPT homicide class M, checking location...\n\n"
    if not match:
        msg = "\nGPT / Manual Class Mismatch\n" + msg
    state = state._replace(gpt3_prompt=prompt,
                           gpt3_response=response_code,
                           next_event='main')
    return combine_actions(
        action2('print_message', msg),
        action2('wait_enter' if not match else 'no_op')
    ), state

def respond_location_class(state: State) -> RxResp:
    """
    Handle response from GPT for location classification
    """
    response, prompt = state.outputs
    response_text = response.classification.value
    response_code = calc.gpt_location_class_code(response.classification)
    msg = calc.prompt_response(prompt, "\n\n" + response_text)
    manual_class = state.articles[state.next_article]['Status']
    match = calc.is_gpt_location_class_correct(response_code, manual_class)
    record_id = state.articles[state.next_article]['RecordId']
    msg = "Manual class / Final GPT class: " + \
            f"{manual_class} / {response_code}\n" + \
            f"Record Id: {record_id}\n\n"
    if not match:
        msg = "\nGPT / Manual Class Mismatch\n" + msg
    state = state._replace(gpt3_prompt=prompt,
                           gpt3_response=response_code,
                           next_event='main')
    return combine_actions(
        action2('print_message', msg),
        action2('wait_enter' if not match else 'no_op')
    ), state
