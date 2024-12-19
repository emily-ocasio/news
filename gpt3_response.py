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
    if state.gpt3_action == 'classify_homicide':
        return respond_homicide_class(state)
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
    msg = f"Manual class: {manual_class}\n" + \
          f"GPT class: {response_text}\n" + \
          f"Record Id: {record_id}\n\n"
    if not match:
        msg = "\nGPT / Manual Class Mismatch\n" + msg
    wait = not match
    if state.next_article == len(state.articles)-1:
        msg += '\nAll articles classified.\n\n'
        wait = True
    state = state._replace(gpt3_prompt=prompt, gpt3_response=response_code)
    return combine_actions(
        action2('print_message', msg),
        action2('wait_enter' if wait else 'no_op'),
        from_reaction(controller.main)
    ), state
