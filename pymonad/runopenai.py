"""
Intent, eliminator and constructors for OpenAI API calls
"""
# pylint:disable=W0212
from dataclasses import dataclass
from collections.abc import Callable
import json
from typing import Any, TypeVar, cast

from jsonref import replace_refs
from openai import OpenAI
from openai.types.responses import Response, ParsedResponse
from openai.types.responses.response_prompt_param import Variables
from pydantic import BaseModel

from .either import Left, Right
from .environment import PromptKey, Environment, EnvKey, all_prompts
from .openai import GPTModel, GPTPrompt, GPTFullResponse
from .run import ErrorPayload, throw, Run, _unhandled, ask, local, pure, \
    put_line

A = TypeVar('A')
P = TypeVar('P', bound=BaseModel)
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
                             variables: dict[str, str],
                             text_format: type[BaseModel],
                             model_key: EnvKey) \
    -> Run[Response | ParsedResponse]:
    """
    Resolve the GPT prompt from the environment.
    """
    var_dict = {k: cast(Variables, v) for k, v in variables.items()}
    def resolve_prompt(env: Environment) -> GPTPrompt:
        gpt_prompts = all_prompts(env)['gpt_prompts']
        if prompt_key not in gpt_prompts:
            throw(ErrorPayload(f"Undefined GPT prompt: {prompt_key}"))
        return GPTPrompt(gpt_prompts[prompt_key], var_dict)

    def resolve_model(env: Environment) -> GPTModel:
        return env['openai_models'].get(
            model_key, env['openai_default_model'])
    return \
        ask() >> (lambda env:
        gpt_response(resolve_prompt(env), resolve_model(env), text_format)
        )

def response_message(specific: Callable[[BaseModel], str],
                     gpt_full: GPTFullResponse) \
                    ->  Run[GPTFullResponse]:
    """
    Generate a response message from the GPT response.
    """
    match gpt_full:
        case Left():
            return pure(gpt_full)
        case Right(resp):
            return \
                put_line(specific(resp.parsed.output)) ^ \
                put_line(f"GPT Usage: {resp.parsed.usage}") ^ \
                put_line(f"GPT reasoning summary: \n{resp.parsed.reasoning}") ^\
                pure(gpt_full)

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
                            text_format=text_format,
                            reasoning={
                                "effort": "medium",
                                "summary": "detailed"
                            }
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

def with_models(models: dict[EnvKey, GPTModel], prog: Run[A]) -> Run[A]:
    """
    Injects the OpenAI models into the program's environment.
    """
    def modify(env: Environment) -> Environment:
        old_models = env['openai_models']
        new_models = old_models | models
        new = cast(Environment, env | {"openai_models": new_models})
        return new
    return local(modify, prog)

def to_json(text_format: type[BaseModel]) -> str:
    """
    Convert the Pydantic model to JSON schema.
    """
    schema = text_format.model_json_schema(mode='serialization')
    schema['name'] = schema['title']
    schema['strict'] = True
    schema.pop('title')
    schema['schema'] = {
        'type': 'object',
        'properties': schema['properties'],
        'required': schema['required'],
        'additionalProperties': False
    }
    schema.pop('properties')
    schema.pop('type')
    schema.pop('required')
    schema = replace_refs(schema, jsonschema=True, proxies=False)
    schema.pop('$defs')
    return json.dumps(schema, indent=2)
