"""
Dataclasses and related functions for the Reader environment.
"""
from collections.abc import Callable, Mapping
from typing import Any, TypedDict, ReadOnly

from .dispatch import InputPrompt
from .openai import GPTPromptTemplate, GPTModel
from .string import String

class EnvKey(String):
    """
    Represents a key in the environment.
    """

class Namespace(EnvKey):
    """
    Represents a namespace in the environment.
    """

class PromptKey(EnvKey):
    """
    Represents a key for prompts in the environment.
    """

type Prompt = InputPrompt | GPTPromptTemplate
type InputPrompts = Mapping[PromptKey, InputPrompt]
type GPTPrompts = Mapping[PromptKey, GPTPromptTemplate]

class AllPrompts(TypedDict):
    """
    Represents combination of InputPrompts and GPTPrompts in the environment.
    """
    input_prompts: ReadOnly[InputPrompts]
    gpt_prompts: ReadOnly[GPTPrompts]

def empty_all_prompts() -> AllPrompts:
    """
    Returns an empty AllPrompts object.
    """
    return {
        "input_prompts": {},
        "gpt_prompts": {}
    }

type NamedPrompts = dict[Namespace, AllPrompts]

class Environment(TypedDict):
    """
    Represents the environment data provided by the reader
    """
    prompt_ns: ReadOnly[Namespace]
    prompts_by_ns: ReadOnly[NamedPrompts]
    db_path: ReadOnly[str]
    openai_client: ReadOnly[Callable]
    openai_default_model: ReadOnly[GPTModel]
    openai_models: ReadOnly[Mapping[EnvKey, GPTModel]]
    extras: ReadOnly[Mapping[EnvKey, Any]]

def all_prompts(env: Environment, ns: Namespace | None = None) -> AllPrompts:
    """
    Returns all prompts from the environment, based on the current namespace.
    """
    if ns is None:
        ns = env['prompt_ns']
    return env['prompts_by_ns'].get(ns, empty_all_prompts())

def to_prompts(prompts_dict: Mapping[str, str | tuple[str, str] | tuple[str,]])\
    -> AllPrompts:
    """
    Converts a dictionary of string prompts to an AllPrompts object.
    """
    def _to_prompt(prompt: str | tuple[str, str] | tuple[str,]) -> Prompt:
        match prompt:
            case (prompt_id, version):
                return GPTPromptTemplate(prompt_id, version)
            case (prompt_id,):
                return GPTPromptTemplate(prompt_id)
            case pr:
                return InputPrompt(str(pr))
    converted = {
        PromptKey(k): _to_prompt(v)
                for k, v in prompts_dict.items()
    }
    input_prompts = {k: v for k, v in converted.items()
                     if isinstance(v, InputPrompt)}
    gpt_prompts = {k: v for k, v in converted.items()
                   if isinstance(v, GPTPromptTemplate)}
    return {
        "input_prompts": input_prompts,
        "gpt_prompts": gpt_prompts
    }
