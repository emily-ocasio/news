"""
Secondary filtering of articles using GPT
"""

from typing import cast

import json
from jsonref import replace_refs
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Either, Right, \
    GPTModel, with_models, response_with_gpt_prompt, foldm_either_loop_bind, \
    wal, reasoning_summary
from menuprompts import input_number, NextStep
from article import Article, Articles, from_rows
from calculations import articles_to_filter_sql
from state import HomicideClassResponse

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "filternumber": "Enter number of articles to further filter via GPT: ",
    "homicide_filter": (
        "pmpt_68b39d0434d88190b0cffa9020bf4d9f0812d04c667840da",)
}
GPT_MODELS = {
    EnvKey("filter"): GPTModel.GPT_5_NANO
}

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
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

        def _filter_next(articles: Articles, next_article: Article) \
            -> Run[Either[str, Articles]]:
            variables = {"article_title": next_article.title,
                         "article_text": next_article.full_text}
            return \
                put_line(f"Filtering article:\n {next_article}") ^ \
                response_with_gpt_prompt(
                    PromptKey("homicide_filter"),
                    variables,
                    HomicideClassResponse,
                    EnvKey("filter")
                ) >> (lambda _resp: wal(
                resp:= cast(ParsedResponse[HomicideClassResponse], _resp),
                parsed:= cast(HomicideClassResponse, resp.output_parsed),
                outputs:= resp.output,
                put_line(f"GPT response: {parsed.classification}\n\n" +
                        f"GPT usage: {resp.usage}\n" +
                        f"reasoning summary: \n{reasoning_summary(resp)}") ^
                pure(Right(Articles.snoc(articles, next_article)))
                ))

        return \
            input_number(PromptKey("filternumber")) >> (lambda num:
            pure(NextStep.CONTINUE) if num <= 0
            else
            put_line(f"Filtering {num} articles...\nJSON schema: \n" +
                     to_json(HomicideClassResponse)) ^
            (from_rows & sql_query(SQL(articles_to_filter_sql()),
                      SQLParams((num,)))) >> (lambda articles:
            foldm_either_loop_bind(articles,
                                   Articles(()),
                                   _filter_next) >> (lambda filtered:
            put_line("All articles filtered.\n") ^ \
            pure(NextStep.CONTINUE)
            )))

    return with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
