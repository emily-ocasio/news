"""
Initial Run monad for application
"""
from pymonad import ask, put_line, Namespace, String, \
    Run, with_namespace, set_, to_prompts
from appstate import user_name

INITIAL_PROMPTS = {
    "name": "Please enter your name:"
}

def initialize_program() -> Run[None]:
    """
    Initialize the program with the given environment.
    """
    def initialize() -> Run[None]:
        return ask() >> (lambda env: (
            put_line("Welcome to the application!") ^
            put_line(
                f"Active publication: "
                f"{env['publication_profile'].session_label}"
            ) ^
            # input_with_prompt(PromptKey("name")) >> (lambda name: \
            # put_line(f"Hello, {name}!") ^
            set_(user_name, String("Emily")) # Default to Emily for now
            ))
    return with_namespace(Namespace(""), to_prompts(INITIAL_PROMPTS),
                          initialize())
