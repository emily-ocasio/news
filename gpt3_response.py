"""
Reactions to handle the GPT-3 responses
"""
from state import RxResp, State
from actionutil import action2, combine_actions, from_reaction
import choose
import calculations as calc


def respond(state: State) -> RxResp:
    """
    Handle response from GPT-3
    """
    response, msg = state.outputs
    msg = calc.prompt_response(msg, response)
    return combine_actions(
        action2('print_message', msg),
        from_reaction(choose.assign_choice)
    ), state
