"""
Secondary filtering of articles using GPtT
"""

from collections.abc import Callable
from functools import partial
from typing import cast, Any

from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Either, Right, \
    GPTModel, with_models, response_with_gpt_prompt, foldm_either_loop_bind, \
    to_gpt_tuple, response_message, to_json, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
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

print_gpt_response = partial(response_message, \
        lambda parsed_output: cast(FormatType, parsed_output).result_str)

def input_number_to_filter() -> Run[int]:
    """
    Prompt the user to input the number of articles to filter.
    """
    def check_if_zero(num: int) -> Run[int]:
        if num < 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(num)
    def after_input(num: int) -> Run[int]:
        return \
            put_line(f"Filtering {num} articles...\n") ^ \
            put_line(f"JSON schema: \n{to_json(FormatType)}") ^ \
            pure(num)
    return \
        input_number(PromptKey("filternumber")) >> check_if_zero >> after_input

def retrieve_articles(num: int) -> Run[Articles]:
    """
    Retrieve the articles to be filtered.
    """
    return \
        from_rows & \
            sql_query(SQL(articles_to_filter_sql()),
                      SQLParams((num,)))

def filter_article(article: Article) -> Run[GPTFullResponse]:
    """
    Filter a single article using GPT
    """
    variables = {"article_title": article.title,
                    "article_text": article.full_text,
                    "article_date": article.full_date}
    return \
        to_gpt_tuple & response_with_gpt_prompt(
            PromptKey("analysis_filter_dc"),
            variables,
            FormatType,
            EnvKey("filter")
        )
type NewGptClass = String | None
type ArticleClassTuple = Tuple[Article, NewGptClass]

def save_article_class(article_class_t: ArticleClassTuple,
                       resp_t: GPTResponseTuple) \
    -> Run[GPTResponseTuple]:
    """
    Save the article classification to the database.
    """
    record_id = article_class_t.fst.record_id
    gpt_class = article_class_t.snd
    return \
        sql_exec(SQL(gpt_homicide_class_sql()),
            SQLParams((gpt_class, record_id))) ^ \
        put_line(f"New GPT class saved: {gpt_class}\n") ^ \
        pure(resp_t)


def save_json(article_class_t: ArticleClassTuple,
              resp_t: GPTResponseTuple) \
    -> Run[GPTResponseTuple]:
    """
    Save the incident JSON to the database if homicide class is 'M'
    """
    record_id = article_class_t.fst.record_id
    gpt_class = article_class_t.snd
    if gpt_class != 'M':
        return pure(resp_t)
    print(cast(FormatType, resp_t.parsed.output).incidents_json)
    return \
        sql_exec(SQL(gpt_victims_sql()),
            SQLParams((
                String(cast(FormatType, resp_t.parsed.output).incidents_json),
                record_id
            ))) ^ \
        put_line("Saved new GPT Incident JSON:\n") ^ \
        pure(resp_t)

def save_filtered_article(article: Article, resp_t: GPTResponseTuple) \
    -> Run[GPTFullResponse]:
    """
    Save and refresh the article information.
    """
    gpt_class = Article.new_gpt_class(resp_t.parsed.output)
    save_class = partial(save_article_class, Tuple(article, gpt_class))
    save_new_json = partial(save_json, Tuple(article, gpt_class))
    return \
        save_class(resp_t) >> save_new_json >> (lambda resp_t: \
        pure(Right(resp_t))
        )

def save_article_fn(article: Article)\
    -> Callable[[Any], Run[Any]]:
    """
    Return a monadic function that saves the filtered article.
    """
    return partial(from_either, partial(save_filtered_article, article))

def process_append(articles: Articles, article: Article) \
    -> Run[Either[str, Articles]]:
    """
    Filter the next article and append to accumulated list
    of processed articles
    """
    append_article= partial(from_either, lambda _: \
                pure(Right(Articles.snoc(articles, article))))
    save_article= save_article_fn(article)
    return \
        put_line(f"Processing article {article}...\n") ^ \
        filter_article(article) >> \
        print_gpt_response >> \
        save_article >> \
        append_article

def process_articles(articles: Articles) -> Run[Either]:
    """
    Process the articles to be filtered.
    """
    return \
        foldm_either_loop_bind(articles, Articles(()), process_append) >> \
        rethrow

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        return \
            input_number_to_filter() >> \
            retrieve_articles >> \
            process_articles ^ \
            put_line("All articles filtered.\n") ^ \
            pure(NextStep.CONTINUE)

    return with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
