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
    state = state._replace(gpt3_prompt = prompt, gpt3_response = response,
                            refresh_article = True)
    if state.gpt3_action in ('extract','small_extract'):
        state = state._replace(extract = calc.remove_quotes(response))
        reaction = save.gpt3_extract
    elif state.gpt3_action == 'humanize':
        state = state._replace(
            humanizing=calc.humanizing_from_response(response, 'word'))
        reaction = save.gpt3_humanize
    else:
        raise Exception('Unknown GPT action')
    return combine_actions(
        action2('print_message', msg),
        from_reaction(reaction),
        action2('no_op' if state.main_flow == 'humanize' else 'wait_enter'),
        from_reaction(controller.main if state.main_flow == 'humanize'
                        else controller.refresh_article)
    ), state
