"""
Intent, eliminator and constructors for OpenAI API calls
"""
# pylint:disable=W0212
from dataclasses import dataclass
from collections.abc import Callable
import json
import threading
import time
from typing import Any, TypeVar, cast, Literal


from jsonref import replace_refs
from openai import OpenAI
from openai.types.responses import Response, ParsedResponse
from openai.types.responses.response_prompt_param import Variables
from pydantic import BaseModel

from .either import Left, Right
from .environment import PromptKey, Environment, EnvKey, all_prompts
from .openai import GPTModel, GPTPrompt, GPTFullResponse, GPTResponseTuple, \
    GPTPromptTemplate
from .run import ErrorPayload, throw, Run, _unhandled, ask, local, pure, \
    put_line, UserAbort

A = TypeVar('A')
P = TypeVar('P', bound=BaseModel)

def _with_elapsed_timer(fn: Callable[[], A]) -> A:
    """
    Run fn while updating an elapsed timer in the console once per second.
    """
    stop = threading.Event()
    start = time.monotonic()

    def tick() -> None:
        while not stop.wait(1):
            elapsed = int(time.monotonic() - start)
            minutes, seconds = divmod(elapsed, 60)
            print(
                f"\rElapsed time: {minutes:02d}:{seconds:02d}",
                end="",
                flush=True,
            )

    thread = threading.Thread(target=tick, daemon=True)
    print("\rElapsed time: 00:00", end="", flush=True)
    thread.start()
    try:
        return fn()
    except KeyboardInterrupt as ex:
        raise UserAbort() from ex
    finally:
        stop.set()
        thread.join()
        elapsed = int(time.monotonic() - start)
        minutes, seconds = divmod(elapsed, 60)
        print(f"\rElapsed time: {minutes:02d}:{seconds:02d}")

@dataclass(frozen=True)
class OAChat:
    """OpenAI Response API call intent"""
    prompt: GPTPrompt
    model: GPTModel
    text_format: type[BaseModel] | None = None
    temperature: float = 0
    effort: Literal['low', 'medium', 'high'] = "low"
    stream: bool = False


def gpt_response(
    prompt: GPTPrompt,
    model: GPTModel,
    text_format: type[BaseModel] | None,
    temperature: float = 0,
    effort: Literal["low", "medium", "high"] = "low",
    stream: bool = False
    ) -> Run[Response | ParsedResponse[BaseModel]]:
    """
    Call the OpenAI API with the given parameters and return the response.
    """
    intent = OAChat(
        prompt, text_format=text_format, model=model, temperature=temperature, \
              effort=effort, stream=stream)
    return Run(lambda self: self._perform(intent, self), _unhandled)

def resolve_prompt_template(env: Environment, prompt_key: PromptKey) \
    -> Run[GPTPromptTemplate]:
    """
    Resolve the GPT prompt template from the environment.
    """
    gpt_prompts = all_prompts(env)['gpt_prompts']
    if prompt_key not in gpt_prompts:
        return throw(ErrorPayload(f"Undefined GPT prompt: {prompt_key}"))
    return pure(gpt_prompts[prompt_key])

def response_with_gpt_prompt(
    prompt_key: PromptKey,
    variables: dict[str, str | None],
    text_format: type[BaseModel],
    model_key: EnvKey,
    effort: Literal["low", "medium", "high"] = "low",
    stream: bool = False
    ) -> Run[Response | ParsedResponse]:
    """
    Resolve the GPT prompt from the environment.
    """
    if any(v is None for v in variables.values()):
        return throw(ErrorPayload(f"Missing variables in prompt: {prompt_key}"))
    var_dict = {k: cast(Variables, v) for k, v in variables.items()}

    def resolve_prompt(env: Environment) -> Run[GPTPrompt]:
        return \
            resolve_prompt_template(env, prompt_key) >> (lambda template: \
            put_line(f"Resolved GPT prompt template for {prompt_key}:\n{template}") ^ \
            pure(GPTPrompt(template, var_dict)))

    def resolve_model(env: Environment) -> GPTModel:
        return env['openai_models'].get(
            model_key, env['openai_default_model'])

    # ask for env, then resolve_prompt (monadic), then call gpt_response
    return \
        ask() >> (lambda env:
        resolve_prompt(env) >> (lambda prompt: \
        gpt_response(
            prompt, resolve_model(env), text_format, effort=effort, stream=stream)
        ))

def _stream_response(
    client: OpenAI,
    model: GPTModel,
    prompt: GPTPrompt,
    effort: Literal["low", "medium", "high"],
) -> Response:
    """
    Stream a response and return the final response.
    """
    with client.responses.stream(
        model=model.value,
        prompt=prompt.to_gpt,
        input=[
            {"role": "user", "content": "Analyze the article."}
        ],
        reasoning={
            "effort": effort,
            "summary": "detailed"
        },
        timeout=300.0
    ) as stream:
        for event in stream:
            if event.type == "response.refusal.delta":
                print(event.delta, end="")
            elif event.type == "response.reasoning_summary_text.delta":
                print(event.delta, end="")
            elif event.type == "response.error":
                print(event.error, end="")
            elif event.type == "response.completed":
                print("\nCompleted\n")
                # print(event.response.output)
        return stream.get_final_response()

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
                case OAChat(prompt, model, text_format, temperature, effort, do_stream):
                    if text_format:
                        if do_stream:
                            ### - DO NOT USE THIS - ###
                            ## API does not properly parse the object at the end ###
                            return _stream_response(client, model, prompt, effort)
                        return _with_elapsed_timer(lambda: client.responses.parse(
                            model=model.value,
                            prompt=prompt.to_gpt,
                            text_format=text_format,
                            reasoning={
                                "effort": effort,
                                "summary": "auto",
                            },
                            timeout=300.0
                        ))
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
    schema.pop('$defs') #type:ignore
    return json.dumps(schema, indent=2)

def from_either(fn: Callable[[GPTResponseTuple], Run[Any]],
                gpt_full: GPTFullResponse) -> Run[Any]:
    """
    Convert a function that operates on a GPTResponseTuple into a Run[Any].
    """
    match gpt_full:
        case Left():
            return pure(gpt_full)
        case Right(resp):
            return fn(resp)

def response_message(specific: Callable[[BaseModel], str],
                     gpt_full: GPTFullResponse) \
                    ->  Run[GPTFullResponse]:
    """
    Generate a response message from the GPT response.
    """
    def message(resp: GPTResponseTuple) -> Run[GPTFullResponse]:
        return \
            put_line(f"{resp.parsed.usage}") ^ \
            put_line(f"GPT reasoning summary: \n{resp.parsed.reasoning}") ^\
            put_line(specific(resp.parsed.output)) ^ \
            pure(gpt_full)
    return from_either(message, gpt_full)
