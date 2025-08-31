"""
Intent, eliminator and constructors for OpenAI API calls
"""
# pylint:disable=W0212
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from openai import OpenAI
from openai.types.responses import Response, ParsedResponse
from openai.types.responses.response_prompt_param import Variables

from pydantic import BaseModel

from .environment import PromptKey, Environment, EnvKey, all_prompts
from .openai import GPTModel, GPTPrompt
from .run import ErrorPayload, throw, Run, _unhandled, ask

A = TypeVar('A')

@dataclass(frozen=True)
class OAChat:
    """OpenAI Response API call intent"""
    prompt: GPTPrompt
    model: GPTModel
    text_format: type[BaseModel] | None = None
    temperature: float = 0

def gpt_response(prompt: GPTPrompt,
                 model: GPTModel,
                 text_format: type[BaseModel] | None,
                 temperature: float = 0) \
                    -> Run[Response | ParsedResponse[BaseModel]]:
    """
    Call the OpenAI API with the given parameters and return the response.
    """
    intent = OAChat(
        prompt, text_format=text_format, model=model, temperature=temperature)
    return Run(lambda self: self._perform(intent, self), _unhandled)

def response_with_gpt_prompt(prompt_key: PromptKey,
                             variables: dict[str, Variables],
                             text_format: type[BaseModel],
                             model_key: EnvKey) \
    -> Run[Response | ParsedResponse]:
    """
    Resolve the GPT prompt from the environment.
    """
    def resolve_prompt(env: Environment) -> GPTPrompt:
        gpt_prompts = all_prompts(env)['gpt_prompts']
        if prompt_key not in gpt_prompts:
            throw(ErrorPayload(f"Undefined GPT prompt: {prompt_key}"))
        return GPTPrompt(gpt_prompts[prompt_key], variables)

    def resolve_model(env: Environment) -> GPTModel:
        return env['openai_models'].get(
            model_key, env['openai_default_model'])
    return \
        ask() >> (lambda env:
        gpt_response(resolve_prompt(env), resolve_model(env), text_format)
        )

def run_openai(
    client_ctor: Callable[[], Any],
    prog: Run[A],
) -> Run[A]:
    """
    Eliminator of OpenAI API calls
    """
    def step(self_run: "Run[Any]") -> A:
        parent = self_run._perform
        client: OpenAI = client_ctor()

        def perform(intent: Any, current: \
                    "Run[Response | ParsedResponse[BaseModel]]") \
            -> Any:
            match intent:
                case OAChat(prompt, model, text_format, temperature):
                    if text_format:
                        return client.responses.parse(
                            model=model.value,
                            prompt=prompt.to_gpt,
                            temperature=temperature,
                            text_format=text_format
                        )
                    return client.responses.create(
                        model=model,
                        prompt=prompt.to_gpt,
                        temperature=temperature
                    )
                case _:
                    return parent(intent, current)

        inner = Run(prog._step, perform)
        return inner._step(inner)
    return Run(step, lambda i, c: c._perform(i, c))
