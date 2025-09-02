"""
Secondary filtering of articles using GPT
"""

from functools import partial
from typing import cast

from pydantic import BaseModel

from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Either, Right, Left, \
    GPTModel, with_models, response_with_gpt_prompt, foldm_either_loop_bind, \
    to_gpt_tuple, response_message, to_json, rethrow
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
    def _specific_message(parsed_output: BaseModel) -> str:
        parsed_output = cast(HomicideClassResponse, parsed_output)
        return f"GPT classification: {parsed_output.classification.value}\n"
    print_specific_message = partial(response_message, _specific_message)
    def _second_filter() -> Run[NextStep]:
        def _filter_next(articles: Articles, next_article: Article) \
            -> Run[Either[str, Articles]]:
            def append_article(gpt_full) -> Run[Either[str, Articles]]:
                if isinstance(gpt_full, Left):
                    return pure(gpt_full)
                return pure(Right(Articles.snoc(articles, next_article)))
            variables = {"article_title": next_article.title,
                         "article_text": next_article.full_text}
            return \
                put_line(f"Filtering article:\n {next_article}") ^ \
                (to_gpt_tuple & response_with_gpt_prompt(
                    PromptKey("homicide_filter"),
                    variables,
                    HomicideClassResponse,
                    EnvKey("filter")
                )) >> print_specific_message >> append_article
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
            rethrow(filtered) ^ \
            put_line("All articles filtered.\n") ^ \
            pure(NextStep.CONTINUE)
            )))

    return with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
