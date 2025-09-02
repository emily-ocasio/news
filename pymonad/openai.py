"""
Dataclasses and related functions for OpenAI API calls
"""
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from openai.types.responses import ResponsePromptParam, Response, ParsedResponse
from openai.types.responses.response_prompt_param import Variables
from pydantic import BaseModel

from .either import Either, Left, Right
from .string import String
from .tuple import Tuple

P = TypeVar("P", bound=BaseModel)
class GPTError(str, Enum):
    """OpenAI GPT model error enumeration"""
    MODEL_NOT_FOUND = "GPT model not found"
    NO_USAGE_DATA = "GPT response has no usage data"
    NOT_PARSED = "GPT response is not parsed"

class GPTModel(str, Enum):
    """OpenAI GPT model enumeration"""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_5_NANO = "gpt-5-nano"

    @classmethod
    def from_string(cls, model_str: str) -> 'GPTModel | None':
        """
        Convert a string to a GPTModel enum member, if possible.
        """
        for model in cls:
            if model_str.startswith(model.value):
                return model
        return None

@dataclass(frozen=True)
class GPTPromptTemplate:
    """OpenAI GPT stored prompt template"""
    id: str
    version: str | None = None

    @property
    def to_dict(self) -> ResponsePromptParam:
        """
        Dictionary representation for use in OpenAI API
        """
        param = ResponsePromptParam(id=self.id)
        if self.version:
            param['version'] = self.version
        return param

@dataclass(frozen=True)
class GPTPrompt:
    """OpenAI GPT instantiated prompt from template"""
    prompt: GPTPromptTemplate
    variables: dict[str, Variables] | None = None

    @property
    def to_gpt(self) -> ResponsePromptParam:
        """
        Dictionary representation for use in OpenAI API
        """
        param =  self.prompt.to_dict
        if self.variables:
            param['variables'] = self.variables
        return param

class PlainText(BaseModel):
    """OpenAI GPT plain text response"""
    output_text: str

class GPTReasoning(String):
    """
    Represents the reasoning summary behind a GPT response.
    """

@dataclass(frozen=True)
class GPTUsage:
    """
    Represents the usage statistics for a GPT response.
    """
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    model_used: GPTModel | None

    def __str__(self):
        return ("GPT Usage for this response:\n"
                f"Model: {self.model_used}, \n"
                f"Input Tokens: {self.input_tokens}, \n"
                f"Cached Tokens: {self.cached_tokens}, \n"
                f"Output Tokens: {self.output_tokens}, \n"
                f"Reasoning Tokens: {self.reasoning_tokens}, \n"
                f"Total Tokens: {self.total_tokens}, \n"
                f"Estimated Cost per 1000 responses: ${self.cost:.4f}\n")

    @property
    def cost(self) -> float:
        """
        Estimated cost per 1000 responses based on this usage
        """
        return self.total_tokens * 0.0004

    @classmethod
    def mempty(cls) -> 'GPTUsage':
        """
        Get an GPT usage stats object.
        """
        return GPTUsage(
            input_tokens=0,
            cached_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
            model_used=None
        )

@dataclass(frozen=True)
class GPTResponseParsed:
    """
    Represents parsed out components of a GPT response.
    """
    usage: GPTUsage
    reasoning: GPTReasoning
    output: BaseModel

@dataclass(frozen=True)
class GPTResponseTuple(Tuple[GPTResponseParsed, Response]):
    """
    Represents a tuple of GPT response stats and parsed response.
    """
    @property
    def parsed(self) -> GPTResponseParsed:
        """
        Get the GPT Response Parsed Values.
        """
        return self.fst

    @property
    def response(self) -> Response:
        """
        Get the original GPT Response object.
        """
        return self.snd

type GPTFullResponse = Either[GPTError, GPTResponseTuple]

def reasoning_summary(resp: Response) -> GPTReasoning:
    """
    Generate a reasoning summary from the GPT response.
    """
    outputs = resp.output
    reasoning_items = [item for item in outputs
                       if item.type == "reasoning"]
    summary_text = f"{len(reasoning_items)} reasoning items found:\n"
    for reasoning in reasoning_items:
        for summary in reasoning.summary:
            summary_text += f"{summary.text}\n"
    return GPTReasoning(summary_text)

def to_gpt_tuple(resp: Response) -> GPTFullResponse:
    """
    Lift the GPT response into a full response Either[Tuple].
    """
    if (model:=GPTModel.from_string(resp.model)) is None:
        return Left(GPTError.MODEL_NOT_FOUND)
    if (usage:=resp.usage) is None:
        return Left(GPTError.NO_USAGE_DATA)
    gpt_usage = GPTUsage(
        input_tokens=usage.input_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        total_tokens=usage.total_tokens,
        model_used=GPTModel(model)
    )
    match resp:
        case ParsedResponse():
            output = resp.output_parsed
            if output is None:
                return Left(GPTError.NOT_PARSED)
        case Response():
            output = PlainText(output_text = resp.output_text)
    parsed = GPTResponseParsed(
        usage=gpt_usage,
        reasoning=reasoning_summary(resp),
        output=output
    )
    return Right.pure(GPTResponseTuple(parsed, resp))
