import deco
from deco import State

def action(name, *args, **kwargs):
    def return_action(state: State) -> State:
        state = state._replace(inputargs = args, inputkwargs = kwargs)
        return getattr(deco, name)(state)
    return return_action
    
state = State()
act = action('act1',1,2,x=10,y=20)
print(f"Final result: {act(state)}")

act_2 = action('act2',arg1=7)
print(f"Final result: {act_2(state)}")