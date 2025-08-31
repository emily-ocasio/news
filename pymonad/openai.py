"""
Dataclasses and related functions for OpenAI API calls
"""
from dataclasses import dataclass
from enum import Enum

from openai.types.responses import ResponsePromptParam
from openai.types.responses.response_prompt_param import Variables

class GPTModel(str, Enum):
    """OpenAI GPT model enumeration"""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_5_NANO = "gpt-5-nano"

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
