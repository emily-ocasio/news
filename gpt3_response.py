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
    state = state._replace(humanizing=calc.humanizing_from_response(response),
                            gpt3_prompt = prompt, gpt3_response = response)
    return combine_actions(
        action2('print_message', msg),
        from_reaction(save.gpt3_humanize),
        action2('wait_enter'),
        from_reaction(controller.refresh_article)
    ), state
