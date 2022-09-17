"""
Reactions to handle the GPT-3 responses
"""
from state import RxResp, State
from actionutil import action2, combine_actions, from_reaction
import save
import controller
import calculations as calc


def respond(state: State) -> RxResp:
    """
    Handle response from GPT-3
    """
    response, prompt = state.outputs
    msg = calc.prompt_response(prompt, response)
    state = state._replace(gpt3_prompt = prompt, gpt3_response = response)
    if state.gpt3_action == 'extract':
        state = state._replace(extract = response)
        reaction = save.gpt3_extract
    elif state.gpt3_action == 'humanize':
        state = state._replace(
            humanizing=calc.humanizing_from_response(response))
        reaction = save.gpt3_humanize
    else:
        raise Exception('Unknown GPT action')
    return combine_actions(
        action2('print_message', msg),
        from_reaction(reaction),
        action2('wait_enter'),
        from_reaction(controller.refresh_article)
    ), state