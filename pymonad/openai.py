"""
Dataclasses and related functions for OpenAI API calls
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Any

from openai.types.responses import ResponsePromptParam, Response, ParsedResponse
from openai.types.responses.response_prompt_param import Variables
from pydantic import BaseModel

from .array import Array
from .either import Either, Left, Right
from .maybe import Maybe, Just, Nothing
from .string import String
from .tuple import Tuple

P = TypeVar("P", bound=BaseModel)
class GPTError(str, Enum):
    """OpenAI GPT model error enumeration"""
    MODEL_NOT_FOUND = "GPT model not found"
    NO_USAGE_DATA = "GPT response has no usage data"
    NOT_PARSED = "GPT response is not parsed"

class GPTTokenType(str, Enum):
    """OpenAI GPT token type enumeration"""
    UNCACHED = "uncached"
    CACHED = "cached"
    OUTPUT = "output"
class GPTModel(str, Enum):
    """OpenAI GPT model enumeration"""
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"

    @classmethod
    def from_string(cls, model_str: str) -> 'GPTModel | None':
        """
        Convert a string to a GPTModel enum member, if possible.
        """
        for model in cls:
            if model_str.startswith(model.value):
                return model
        return None

    def price(self, token_type: GPTTokenType) -> float:
        """
        Get the price per 1M tokens for a specific token type.
        """
        match self, token_type:
            case GPTModel.GPT_5_NANO, GPTTokenType.UNCACHED:
                return 0.05
            case GPTModel.GPT_5_NANO, GPTTokenType.CACHED:
                return 0.005
            case GPTModel.GPT_5_NANO, GPTTokenType.OUTPUT:
                return 0.4
            case GPTModel.GPT_5_MINI, GPTTokenType.UNCACHED:
                return 0.025
            case GPTModel.GPT_5_MINI, GPTTokenType.CACHED:
                return 0.25
            case GPTModel.GPT_5_MINI, GPTTokenType.OUTPUT:
                return 2.0
            case _:
                raise ValueError(
                    f"Unknown model or token type: {self}, {token_type}")

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
    @classmethod
    def from_row(cls, row: Any) -> "GPTReasoning":
        """
        Convert a gptResults row to GPTReasoning.
        """
        return cls(row["Reasoning"] or "")

def gpt_usage_reasoning_from_rows(
    rows: Array
) -> Maybe[Tuple[GPTUsage, GPTReasoning]]:
    """
    Convert the latest gptResults row to a Maybe of usage + reasoning.
    """
    if len(rows) == 0:
        return Nothing
    row = rows[0]
    return Just(Tuple(GPTUsage.from_row(row), GPTReasoning.from_row(row)))

BUNDLE = 1000
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
            f"Model: {self.model_used.value if self.model_used else 'None'}, \n"
            f"Uncached Tokens:  {self.uncached_tokens:5d} \n"
            f"Cached Tokens:    {self.cached_tokens:5d} \n"
            f"      Total Input Tokens:  {self.input_tokens:5d} \n"
            f"Message Tokens:   {self.actual_output_tokens:5d} \n"
            f"Reasoning Tokens: {self.reasoning_tokens:5d} \n"
            f"      Total Output Tokens: {self.output_tokens:5d} \n"
            f"      Total Tokens:        {self.total_tokens:5d} \n"
            f"Estimated Cost per {BUNDLE} responses: "
            f"${self.cost(BUNDLE):.4f}\n")

    def cost(self, bundle = 1000) -> float:
        """
        Estimated cost per bundle of responses based on this usage
        """
        ratio = bundle / 1_000_000
        if self.model_used is None:
            return 0.0
        unc_price = self.model_used.price(GPTTokenType.UNCACHED)
        cached_price = self.model_used.price(GPTTokenType.CACHED)
        output_price = self.model_used.price(GPTTokenType.OUTPUT)
        return ratio * (unc_price * self.uncached_tokens \
            + cached_price * self.cached_tokens \
            + output_price * self.output_tokens)

    @property
    def uncached_tokens(self) -> int:
        """
        Uncached input tokens used in the response.
        """
        return self.input_tokens - self.cached_tokens

    @property
    def actual_output_tokens(self) -> int:
        """
        Actual (non-reasoning)output tokens used in the response.
        """
        return self.output_tokens - self.reasoning_tokens

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

    @classmethod
    def from_row(cls, row: Any) -> "GPTUsage":
        """
        Convert a gptResults row to GPTUsage.
        """
        input_tokens = row["TotalInputTokens"] or 0
        cached_tokens = row["CachedInputTokens"] or 0
        output_tokens = row["TotalOutputTokens"] or 0
        reasoning_tokens = row["ReasoningTokens"] or 0
        model_str = row["Model"] or ""
        model = GPTModel.from_string(model_str) if model_str else None
        return cls(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
            model_used=model
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
                print(f"DEBUG: Raw GPT output_text: {resp.output_text}/nFull output: {resp.output}")
                return Left(GPTError.NOT_PARSED)
        case Response():
            output = PlainText(output_text = resp.output_text)
    parsed = GPTResponseParsed(
        usage=gpt_usage,
        reasoning=reasoning_summary(resp),
        output=output
    )
    return Right.pure(GPTResponseTuple(parsed, resp))
