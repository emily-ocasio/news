"""
Initial Run monad for application
"""
from pymonad import put_line, Namespace, \
    PromptKey, Run, with_namespace, input_with_prompt, set_, to_prompts
from appstate import user_name

INITIAL_PROMPTS = {
    "name": "Please enter your name:"
}

def initialize_program() -> Run[None]:
    """
    Initialize the program with the given environment.
    """
    def initialize() -> Run[None]:
        return \
            put_line("Welcome to the application!") ^ \
            input_with_prompt(PromptKey("name")) >> (lambda name: \
            put_line(f"Hello, {name}!") ^
            set_(user_name, name)
            )

    return with_namespace(Namespace(""), initialize(),
                          prompts=to_prompts(INITIAL_PROMPTS))
