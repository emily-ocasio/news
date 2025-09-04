"""
Secondary filtering of articles using GPT
"""

from functools import partial
from typing import cast

import json
from pydantic import BaseModel

from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Either, Right, \
    GPTModel, with_models, response_with_gpt_prompt, foldm_either_loop_bind, \
    to_gpt_tuple, response_message, to_json, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String
from menuprompts import input_number, NextStep
from article import Article, Articles, from_rows
from calculations import articles_to_filter_sql, gpt_homicide_class_sql, \
    gpt_victims_sql
from state import WashingtonPostArticleAnalysis

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "filternumber": "Enter number of articles to further filter via GPT: ",
    "homicide_filter": (
        "pmpt_68b39d0434d88190b0cffa9020bf4d9f0812d04c667840da",),
    "homicide_filter_dc": (
        "pmpt_68b74e67a7348196b249502a0b15992c0d9f2d89c372f7a0",),
    "combined_filter_dc": (
        "pmpt_68b86cea7d9c8190a6c87d22bdeed028041d76441e5cbcee",),
    "analysis_filter_dc": (
        'pmpt_68b8a1beb47081959598ebf1f4a9048804c14be36aa3f9a3',)
}
GPT_MODELS = {
    EnvKey("filter"): GPTModel.GPT_5_NANO
}
FormatType = WashingtonPostArticleAnalysis

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _incidents_str(parsed_output: BaseModel) -> str:
        parsed_output = cast(FormatType, parsed_output)
        if len(parsed_output.incidents) == 0:
            return ""
        incidents_list = parsed_output.model_dump()['incidents']
        return json.dumps(incidents_list, indent=2)

    def _specific_message(parsed_output: BaseModel) -> str:
        parsed_output = cast(FormatType, parsed_output)
        art_class = parsed_output.article_classification.value
        hom_class = parsed_output.homicide_classification.value \
            if parsed_output.homicide_classification else 'None'
        incidents = _incidents_str(parsed_output)
        return f"GPT article classification: {art_class}\n" + \
            f"GPT homicide classification: {hom_class}\n" + \
            incidents
    print_specific_message = partial(response_message, _specific_message)

    def _save_article(article: Article, resp_t: GPTResponseTuple) \
        -> Run[GPTFullResponse]:
        """
        Save and refresh the article information.
        """
        def _save_json() -> Run[None]:
            if gpt_class != 'M':
                return pure(None)
            return \
                sql_exec(SQL(gpt_victims_sql()),
                    SQLParams((
                        gpt_json:=String(_incidents_str(resp_t.parsed.output)),
                        article.record_id
                    ))) ^ put_line(f"GPT Incident JSON:\n{gpt_json}")
        return \
            sql_exec(SQL(gpt_homicide_class_sql()),
                SQLParams((
                    gpt_class:=Article.new_gpt_class(resp_t.parsed.output),
                    article.record_id
               ))) ^ \
            _save_json() ^ \
            put_line(f"New GPT class saved: {gpt_class}\n") ^ \
            pure(Right(resp_t))

    def _filter_next(articles: Articles, next_article: Article) \
        -> Run[Either[str, Articles]]:
        variables = {"article_title": next_article.title,
                        "article_text": next_article.full_text,
                        "article_date": next_article.full_date}
        append_article= partial(from_either, lambda _: \
                    pure(Right(Articles.snoc(articles, next_article))))
        save_article=partial(from_either,
                                partial(_save_article, next_article))
        return \
            put_line(f"Filtering article:\n {next_article}\n"
                     f"Date: {next_article.full_date}\n") ^ \
            (to_gpt_tuple & response_with_gpt_prompt(
                PromptKey("analysis_filter_dc"),
                variables,
                FormatType,
                EnvKey("filter")
            )) >> print_specific_message >> save_article >> append_article

    def _second_filter() -> Run[NextStep]:
        return \
            input_number(PromptKey("filternumber")) >> (lambda num:
            pure(NextStep.CONTINUE) if num <= 0
            else
            put_line(f"Filtering {num} articles...\nJSON schema: \n" +
                     to_json(FormatType)) ^
            (from_rows & sql_query(SQL(articles_to_filter_sql()),
                      SQLParams((num,)))) >> (lambda articles:
            foldm_either_loop_bind(articles,
                                   Articles(()),
                                   _filter_next) >>
            rethrow ^ \
            put_line("All articles filtered.\n") ^ \
            pure(NextStep.CONTINUE)
            ))

    return with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
